import base64
import html
import json
import logging
import os
import re
import urllib.parse
from pathlib import Path
from typing import Literal, cast

import streamlit as st
from pydantic import ValidationError
from st_keyup import st_keyup

import sys

src_path = Path(__file__).resolve().parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from backend.drive import (  # noqa: E402
    create_library,
    delete_library,
    download_file,
    get_or_create_root_folder,
    get_papers_folder,
    list_libraries,
    upload_file_to_folder,
    upload_library_index,
    DriveTransientError,
)
from backend.models import LibraryIndex, PaperMetadata  # noqa: E402
from frontend.auth import authenticate_user  # noqa: E402
from frontend.constants import (  # noqa: E402
    DEFAULT_FASTAPI_URL,
    EDITED_PDF_FILENAME,
    GENERATE_METADATA_HELP,
    HF_TOKEN_MISSING_HELP,
    JSON_MIME_TYPE,
    LABEL_TO_STATUS,
    MAX_DUPLICATE_NAMES_TO_LIST,
    META_FILENAME,
    PAPER_ID_PATTERN,
    PDF_FILENAME,
    SIMILAR_FILTER_LABEL,
    STATUS_ICONS,
    STATUS_LABELS,
)
from frontend.downloads import zip_download_filename, zip_marked_pdfs  # noqa: E402
from frontend.library import (  # noqa: E402
    add_tags_to_selected,
    delete_selected_papers,
    init_library_state,
    remove_tags_from_selected,
    sync_library_index,
)
from frontend.library_filters import (  # noqa: E402
    filter_papers,
    get_all_tags,
    get_duplicate_pids,
    get_missing_metadata_pids,
)
from frontend.metadata_generation import (  # noqa: E402
    generate_metadata_for_paper,
    generate_metadata_for_selected,
    sync_paper_metadata,
)
from frontend.text_utils import strip_pdf_suffix  # noqa: E402
from frontend.uploads import upload_papers  # noqa: E402

# The functions/constants below aren't called directly by this module's own
# code anymore (they moved to frontend/auth.py, uploads.py, library.py,
# library_filters.py, and metadata_generation.py) but the test suite patches
# and reads them via `app.<name>` (e.g. `mocker.patch.object(app, "...")`),
# so they're re-exported here rather than dropped as unused imports.
from backend.drive import (  # noqa: E402,F401
    OAUTH_FLOWS as OAUTH_FLOWS,
    PAPERS_DIR as PAPERS_DIR,
    add_oauth_flow as add_oauth_flow,
    create_paper_folder as create_paper_folder,
    delete_paper_folder as delete_paper_folder,
    get_library_index_file as get_library_index_file,
    get_oauth_flow as get_oauth_flow,
    load_credentials_from_file as load_credentials_from_file,
    save_credentials as save_credentials,
)
from backend.huggingface_client import (  # noqa: E402,F401
    DEFAULT_EMBEDDING_MODEL as DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL as DEFAULT_GENERATION_MODEL,
    HFTokenMissingError as HFTokenMissingError,
    embed_text as embed_text,
    extract_pdf_text as extract_pdf_text,
    find_similar_papers as find_similar_papers,
    generate_paper_metadata as generate_paper_metadata,
    is_hf_token_configured,
)
from backend.host_check import get_non_loopback_host_warning  # noqa: E402
from frontend.branding import apply_branding  # noqa: E402
from frontend.constants import (  # noqa: E402,F401
    BULK_GENERATE_DELAY_SECONDS as BULK_GENERATE_DELAY_SECONDS,
)
from frontend.metadata_generation import (  # noqa: E402,F401
    persist_generated_metadata as persist_generated_metadata,
)


logger = logging.getLogger(__name__)

_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "paper-butler-logo.svg"


@st.cache_data
def _load_logo_base64(path: Path) -> str:
    """Reads and base64-encodes the logo SVG, cached across reruns.

    Args:
        path: Filesystem path to the logo SVG.

    Returns:
        The base64-encoded SVG content as an ASCII string.
    """
    return base64.b64encode(path.read_bytes()).decode("ascii")


st.set_page_config(
    layout="wide",
    page_title="Paper Butler",
    page_icon=str(_LOGO_PATH) if _LOGO_PATH.exists() else "📚",
    initial_sidebar_state="expanded",
)
apply_branding()
if _LOGO_PATH.exists():
    st.logo(str(_LOGO_PATH))

# Session-state keys backing the icon bar's mutually exclusive staged
# actions - only one of delete/generate/add-tag/remove-tag can be staged at
# a time, so arming one clears the others rather than stacking their forms.
BULK_ACTION_STATE_KEYS = (
    "confirm_delete_pids",
    "confirm_generate_pids",
    "show_add_tag_pids",
    "show_remove_tag_pids",
    "show_download_pids",
)


def _stage_bulk_action(state_key: str, pids: list[str]) -> None:
    """Stages a bulk action for the icon bar, clearing any other staged action.

    Args:
        state_key: The `st.session_state` key to stage `pids` under - one of
            `BULK_ACTION_STATE_KEYS`.
        pids: The paper IDs the staged action applies to.
    """
    for key in BULK_ACTION_STATE_KEYS:
        if key != state_key:
            st.session_state.pop(key, None)
    st.session_state[state_key] = pids


def _clear_bulk_actions() -> None:
    """Clears any staged icon-bar action without staging a new one."""
    for key in BULK_ACTION_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop("download_zip_result", None)


def _toggle_select_all(filtered_pids: list[str]) -> None:
    """Marks or clears every paper checkbox to match the "Mark all" toggle.

    Every paper's checkbox is cleared first, then - only when the toggle was
    switched on - re-marked for exactly `filtered_pids`. Always starting from
    a clean slate keeps this scoped to the papers currently matching the
    active search/filter, with no stale marks surviving from a prior
    selection made under a different filter.

    Each checkbox is reset via an explicit `False` assignment rather than
    `pop()`-ing its key: Streamlit's frontend only re-renders a checkbox
    widget when it receives an explicit value for that key, so deleting the
    key left the browser showing the widget's last-rendered (checked) state
    even though the backend no longer had it selected.

    Args:
        filtered_pids: The paper IDs currently satisfying the active
            search/filter criteria, captured at the time the toggle was
            rendered.
    """
    mark_all = st.session_state.select_all_toggle
    for pid in st.session_state.index.papers:
        st.session_state[f"chk_{pid}"] = False
    if mark_all:
        for pid in filtered_pids:
            st.session_state[f"chk_{pid}"] = True


def _reset_selected_paper() -> None:
    """Clears the selected paper, returning the app to the library view."""
    st.session_state.selected_paper = None
    _clear_bulk_actions()


def _render_logo_reset_button() -> None:
    """Overlays an invisible button on the top-left logo to reset paper view.

    `st.logo` has no click callback, so a transparent Streamlit button is
    positioned on top of it via CSS, letting a click on the logo reset the
    selected paper without a full page reload (which would drop session
    state like the active library).
    """
    st.markdown(
        "<style>"
        ".st-key-reset_logo_btn { position: fixed; top: 0; left: 0; "
        "width: 14rem; height: 3.5rem; z-index: 999999; }"
        ".st-key-reset_logo_btn button { width: 100%; height: 100%; "
        "opacity: 0; cursor: pointer; }"
        "</style>",
        unsafe_allow_html=True,
    )
    if st.button("Reset selected paper", key="reset_logo_btn"):
        _reset_selected_paper()
        st.rerun()


def _uncheck_mark_all_if_unmarked(pid: str) -> None:
    """Clears the "Mark all" toggle when a paper's own checkbox is unmarked.

    Individually unchecking one paper means the checked set no longer
    covers every filtered paper, so "Mark all" can no longer honestly read
    as checked - without this, it stayed visually marked after such an
    uncheck even though the underlying selection was no longer "all".

    Args:
        pid: The paper ID whose checkbox just changed.
    """
    if not st.session_state.get(f"chk_{pid}"):
        st.session_state.select_all_toggle = False


def main() -> None:
    """The main entry point for the Streamlit frontend application.

    Authenticates the user, displays the library selection UI, and handles
    all interactions including paper uploads, search, and metadata editing.

    Args:
        None

    Returns:
        None
    """
    creds = authenticate_user()
    if not creds:
        return

    if st.session_state.get("selected_paper"):
        _render_logo_reset_button()

    if not st.session_state.get("selected_paper"):
        _, header_col, _ = st.columns([1, 2, 1])
        with header_col:
            logo_html = ""
            if _LOGO_PATH.exists():
                logo_b64 = _load_logo_base64(_LOGO_PATH)
                logo_html = (
                    f"<img src='data:image/svg+xml;base64,{logo_b64}' "
                    "width='200' style='display: block;' />"
                )
            st.markdown(
                "<div style='display: flex; justify-content: center; "
                "align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>"
                f"{logo_html}"
                "<h1 style='margin: 0;'>Paper Butler</h1>"
                "</div>",
                unsafe_allow_html=True,
            )

    host_warning = get_non_loopback_host_warning()
    if host_warning is not None:
        st.warning(host_warning)

    if "root_id" not in st.session_state:
        st.session_state.root_id = get_or_create_root_folder(creds)

    root_id = st.session_state.root_id

    # Library Selection Screen
    if "current_lib_id" not in st.session_state:
        libraries = list_libraries(creds, root_id)

        if len(libraries) == 1 and not st.session_state.get("manual_library_selection"):
            only_lib = libraries[0]
            papers_id = get_papers_folder(creds, only_lib["id"])
            init_library_state(creds, only_lib["id"], papers_id, only_lib["name"])
            st.rerun()
            return

        st.subheader("Select or Create a Library")

        if not is_hf_token_configured():
            st.info(HF_TOKEN_MISSING_HELP)

        col1, col2 = st.columns(2)
        with col1:
            if libraries:
                lib_options = {lib["id"]: lib["name"] for lib in libraries}
                selected_lib = st.selectbox(
                    "Existing Libraries",
                    options=list(lib_options.keys()),
                    format_func=lambda x: lib_options[x],
                )
                if st.button("Open Library"):
                    papers_id = get_papers_folder(creds, selected_lib)
                    init_library_state(
                        creds, selected_lib, papers_id, lib_options[selected_lib]
                    )
                    st.rerun()
                if (
                    st.session_state.get("confirm_delete_lib_id") is not None
                    and st.session_state.confirm_delete_lib_id != selected_lib
                ):
                    st.session_state.confirm_delete_lib_id = None

                if st.button("Delete Library"):
                    st.session_state.confirm_delete_lib_id = selected_lib

                if st.session_state.get("confirm_delete_lib_id") in lib_options:
                    lib_id_to_delete = st.session_state.confirm_delete_lib_id
                    is_deleting_lib = (
                        st.session_state.get("deleting_lib_id") == lib_id_to_delete
                    )
                    delete_lib_error = st.session_state.pop("delete_lib_error", None)
                    if delete_lib_error is not None:
                        st.error(delete_lib_error)
                    st.warning(
                        f"Delete library '{lib_options[lib_id_to_delete]}' and "
                        "all its papers? This will move the library and all "
                        "its papers to your Google Drive trash."
                    )
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button(
                            "Confirm",
                            key="confirm_delete_lib_btn",
                            disabled=is_deleting_lib,
                        ):
                            st.session_state.deleting_lib_id = lib_id_to_delete
                            st.rerun()
                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key="cancel_delete_lib_btn",
                            disabled=is_deleting_lib,
                        ):
                            st.session_state.confirm_delete_lib_id = None
                            st.rerun()
                    if is_deleting_lib:
                        with st.spinner("Deleting library..."):
                            try:
                                delete_library(creds, lib_id_to_delete)
                            except DriveTransientError:
                                logger.exception("Failed to delete library")
                                st.session_state.delete_lib_error = (
                                    "Google Drive is temporarily unavailable. "
                                    "Please try again in a moment."
                                )
                            except Exception:
                                logger.exception("Failed to delete library")
                                st.session_state.delete_lib_error = (
                                    "Failed to delete library. Please try again."
                                )
                            else:
                                st.session_state.confirm_delete_lib_id = None
                            finally:
                                st.session_state.deleting_lib_id = None
                                st.rerun()
            else:
                st.info("No existing libraries found.")

        with col2:
            new_lib_name = st.text_input("New Library Name")
            if st.button("Create Library") and new_lib_name:
                lib_info = create_library(creds, root_id, new_lib_name)
                init_library_state(
                    creds,
                    lib_info["lib_id"],
                    lib_info["papers_id"],
                    lib_info["lib_name"],
                )
                upload_library_index(creds, lib_info["papers_id"], LibraryIndex())
                st.success(f"Library '{new_lib_name}' created!")
                st.rerun()
        return

    # Library View
    if "index" not in st.session_state:
        with st.spinner("Syncing library..."):
            sync_library_index(creds)

    with st.sidebar:

        def switch_lib() -> None:
            """Clears the current library's session state to return to library selection.

            Also forces the manual selection screen to show even if only one
            library exists, since the user explicitly asked to switch.
            """
            st.session_state.manual_library_selection = True
            for k in [
                "current_lib_id",
                "current_lib_name",
                "current_papers_id",
                "index",
                "last_sync_time",
                "confirm_delete_pids",
                "confirm_generate_pids",
                "show_add_tag_pids",
                "show_remove_tag_pids",
                "show_download_pids",
                "download_zip_result",
            ]:
                st.session_state.pop(k, None)

        # Expander headers have no built-in alignment option, so center
        # them to match the full-width "Switch Library" button above them.
        st.markdown(
            "<style>[data-testid='stExpander'] summary "
            "{ justify-content: center; }</style>",
            unsafe_allow_html=True,
        )

        lib_name = st.session_state.get(
            "current_lib_name", st.session_state.current_lib_id
        )
        st.caption(f"📁 Library: {lib_name}")
        st.button("Switch Library", on_click=switch_lib, use_container_width=True)

        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        upload_flash = st.session_state.pop("upload_flash", None)
        with st.expander("Upload Paper(s)", expanded=upload_flash is not None):
            if upload_flash is not None:
                duplicates_skipped = upload_flash["duplicates_skipped"]
                if len(duplicates_skipped) > MAX_DUPLICATE_NAMES_TO_LIST:
                    st.warning(
                        f"Skipped {len(duplicates_skipped)} duplicate files "
                        "already in this library."
                    )
                else:
                    for skipped_name in duplicates_skipped:
                        st.warning(
                            f"Skipped {skipped_name}: a paper with that title "
                            "already exists in this library."
                        )
                if upload_flash["all_succeeded"]:
                    st.success(upload_flash["message"])
                else:
                    st.warning(upload_flash["message"])

            uploaded_files = st.file_uploader(
                "Choose PDF files",
                type="pdf",
                accept_multiple_files=True,
                key=str(st.session_state.uploader_key),
            )
            if uploaded_files:
                if st.button("Upload"):
                    file_count = len(uploaded_files)
                    progress = st.progress(0.0, text=f"Uploading (0/{file_count})...")

                    def on_progress(i: int, total: int, filename: str) -> None:
                        """Advances the upload progress bar after each file.

                        Args:
                            i (int): The 1-based index of the file just processed.
                            total (int): The total number of files in the batch.
                            filename (str): The name of the file just processed.
                        """
                        progress.progress(
                            i / total, text=f"Uploading ({i}/{total}): {filename}"
                        )

                    try:
                        all_succeeded = upload_papers(
                            creds, uploaded_files, on_progress=on_progress
                        )
                    finally:
                        try:
                            upload_library_index(
                                creds,
                                st.session_state.current_papers_id,
                                st.session_state.index,
                            )
                        finally:
                            progress.empty()
                            st.session_state.last_sync_time = None
                    st.session_state.uploader_key += 1
                    duplicates_skipped = st.session_state.get(
                        "duplicate_uploads_skipped", []
                    )
                    if all_succeeded:
                        message = "Uploaded successfully!"
                        if duplicates_skipped:
                            message = (
                                "Uploaded successfully! (Duplicate files "
                                "were skipped — see warnings above.)"
                            )
                        st.session_state.upload_flash = {
                            "duplicates_skipped": duplicates_skipped,
                            "all_succeeded": True,
                            "message": message,
                        }
                        st.rerun()
                    else:
                        if len(duplicates_skipped) > MAX_DUPLICATE_NAMES_TO_LIST:
                            st.warning(
                                f"Skipped {len(duplicates_skipped)} duplicate "
                                "files already in this library."
                            )
                        else:
                            for skipped_name in duplicates_skipped:
                                st.warning(
                                    f"Skipped {skipped_name}: a paper with "
                                    "that title already exists in this "
                                    "library."
                                )
                        st.warning(
                            "Some files failed to upload. See the errors above; "
                            "re-select the failed files to retry."
                        )

        with st.expander("Library Papers", expanded=True):
            # A paper's checkbox stays checked in session state even while
            # it's hidden by a search/status/tag filter, so scan every
            # known paper (not just the currently filtered ones) to decide
            # whether the bin icon should read as "armed".
            checked_pids = [
                pid
                for pid in st.session_state.index.papers
                if st.session_state.get(f"chk_{pid}")
            ]

            # Both search_box (a custom st_keyup iframe) and the native
            # status_filter/tags_filter multiselects below keep showing
            # their last value if only their session-state key is popped -
            # Streamlit only re-renders a widget when its *key* changes, not
            # its value (see uploader_key above for the same
            # fixed-by-remount pattern applied to the file uploader).
            filter_nonce = st.session_state.get("filter_nonce", 0)
            search_box_key = f"search_box_{filter_nonce}"
            status_filter_key = f"status_filter_{filter_nonce}"
            tags_filter_key = f"tags_filter_{filter_nonce}"

            def clear_filters() -> None:
                """Resets the Search, Status, and Tags filters.

                Bumps ``filter_nonce`` to force the search box and the
                Status/Tags multiselects to remount empty. Popping their
                session-state keys alone doesn't reset a native st.multiselect
                in practice - its frontend component keeps showing the
                last-picked chips across reruns unless its `key` itself
                changes, the same remount-on-nonce pattern the search box
                uses for its st_keyup iframe.
                """
                for k in [search_box_key, status_filter_key, tags_filter_key]:
                    st.session_state.pop(k, None)
                st.session_state.filter_nonce = filter_nonce + 1

            filters_active = bool(
                st.session_state.get(search_box_key)
                or st.session_state.get(status_filter_key)
                or st.session_state.get(tags_filter_key)
            )
            # Narrow columns with no gap keep the two icons adjacent instead
            # of centered in two full-width halves; the trailing column
            # just absorbs the remaining space. All three icons use a fixed
            # "secondary" type; no per-button background-color styling.
            #
            # The [1, 1, 1, 7] ratio is only a *starting* width: Streamlit
            # columns are flex items that shrink with their container, so
            # dragging the sidebar's resize handle narrower than a few
            # hundred px would shrink these three icon columns past the
            # button's natural size, making the buttons overlap and the
            # emoji glyphs spill outside their bounds. Pin the icon columns
            # to a fixed size (flex: 0 0 auto) so only the trailing spacer
            # column shrinks/grows, and clip+center each button's content so
            # its emoji can never render outside the button box.
            #
            # At very narrow sidebar widths the row itself wraps (the three
            # fixed-size columns no longer fit next to the shrunk spacer
            # column), stacking icons onto a second row. A bottom margin on
            # every button keeps that wrapped row from touching the one
            # above it, matching the existing horizontal gap so wrapped
            # icons stay evenly spaced in both directions.
            #
            # Streamlit reserves its own ~1rem gap above the row plus the
            # expander's native top padding, stacking into a large empty
            # band between the "Library Papers" header and the icon row.
            # Zero out the expander's native padding for this row and pull
            # the row up with a negative margin to close most of that gap;
            # a button's `help` tooltip is allowed to land on the header
            # text when hovering a top-row icon — that overlap is accepted
            # in exchange for a tight header-to-icon spacing.
            st.markdown(
                "<style>"
                "div[data-testid='stExpanderDetails']:has(.st-key-trash_icon)"
                "{ padding-top: 0; }"
                "div[data-testid='stHorizontalBlock']:has(.st-key-trash_icon)"
                "{ padding-top: 0; margin-top: -0.6rem; }"
                "div[data-testid='stColumn']:has(.st-key-trash_icon),"
                "div[data-testid='stColumn']:has(.st-key-bulk_generate_icon),"
                "div[data-testid='stColumn']:has(.st-key-generate_missing_icon),"
                "div[data-testid='stColumn']:has(.st-key-add_tag_icon),"
                "div[data-testid='stColumn']:has(.st-key-remove_tag_icon),"
                "div[data-testid='stColumn']:has(.st-key-download_icon),"
                "div[data-testid='stColumn']:has(.st-key-clear_filters_icon)"
                "{ flex: 0 0 auto; width: auto; min-width: 2.5rem; }"
                ".st-key-trash_icon button,"
                ".st-key-bulk_generate_icon button,"
                ".st-key-generate_missing_icon button,"
                ".st-key-add_tag_icon button,"
                ".st-key-remove_tag_icon button,"
                ".st-key-download_icon button,"
                ".st-key-clear_filters_icon button"
                "{ width: 2.5rem; height: 2.5rem; min-width: 2.5rem; padding: 0;"
                " display: flex; align-items: center; justify-content: center;"
                " overflow: hidden; line-height: 1; margin-bottom: 0.4rem; }"
                ".st-key-trash_icon button, .st-key-bulk_generate_icon button,"
                ".st-key-generate_missing_icon button, .st-key-add_tag_icon button,"
                ".st-key-remove_tag_icon button, .st-key-download_icon button"
                "{ margin-right: 0.4rem; }"
                "</style>",
                unsafe_allow_html=True,
            )
            (
                icon_col1,
                icon_col2,
                icon_col3,
                icon_col4,
                icon_col5,
                icon_col6,
                icon_col7,
                _icon_spacer,
            ) = st.columns([1, 1, 1, 1, 1, 1, 1, 4], gap=None)
            # Rendered outside the columns below so a wide message box never
            # stretches the auto-width icon columns and shoves the later
            # icons out of place.
            icon_bar_message: tuple[str, str] | None = None
            with icon_col1:
                if st.button(
                    "🗑️",
                    key="trash_icon",
                    help="Delete selected papers"
                    if checked_pids
                    else "Select papers to delete",
                    type="secondary",
                    disabled=not checked_pids,
                ):
                    _stage_bulk_action("confirm_delete_pids", checked_pids)
            hf_token_configured = is_hf_token_configured()
            with icon_col2:
                if st.button(
                    "✨",
                    key="bulk_generate_icon",
                    disabled=not hf_token_configured or not checked_pids,
                    help=(
                        HF_TOKEN_MISSING_HELP
                        if not hf_token_configured
                        else "Select papers to generate metadata"
                        if not checked_pids
                        else GENERATE_METADATA_HELP
                    ),
                    type="secondary",
                ):
                    _stage_bulk_action("confirm_generate_pids", checked_pids)
            with icon_col3:
                if st.button(
                    "🪄",
                    key="generate_missing_icon",
                    disabled=not hf_token_configured,
                    help=(
                        "Generate metadata for every paper that doesn't have any yet"
                        if hf_token_configured
                        else HF_TOKEN_MISSING_HELP
                    ),
                    type="secondary",
                ):
                    missing_pids = list(
                        get_missing_metadata_pids(st.session_state.index)
                    )
                    if missing_pids:
                        _stage_bulk_action("confirm_generate_pids", missing_pids)
                    else:
                        _clear_bulk_actions()
                        icon_bar_message = ("info", "Every paper already has metadata.")
            with icon_col4:
                if st.button(
                    "🏷️",
                    key="add_tag_icon",
                    help="Add a tag to selected papers"
                    if checked_pids
                    else "Select papers to tag",
                    type="secondary",
                    disabled=not checked_pids,
                ):
                    _stage_bulk_action("show_add_tag_pids", checked_pids)
            with icon_col5:
                if st.button(
                    "🚫",
                    key="remove_tag_icon",
                    help="Remove a tag from selected papers"
                    if checked_pids
                    else "Select papers to untag",
                    type="secondary",
                    disabled=not checked_pids,
                ):
                    _stage_bulk_action("show_remove_tag_pids", checked_pids)
            with icon_col6:
                if st.button(
                    "📥",
                    key="download_icon",
                    help="Download PDFs for selected papers as a zip"
                    if checked_pids
                    else "Select papers to download",
                    type="secondary",
                    disabled=not checked_pids,
                ):
                    _stage_bulk_action("show_download_pids", checked_pids)
                    st.session_state.pop("download_zip_result", None)
            with icon_col7:
                st.button(
                    "🧹",
                    key="clear_filters_icon",
                    help="Clear filters"
                    if filters_active
                    else "No active filters to clear",
                    type="secondary",
                    disabled=not filters_active,
                    on_click=clear_filters,
                )

            if icon_bar_message is not None:
                message_kind, message_text = icon_bar_message
                if message_kind == "warning":
                    st.warning(message_text)
                else:
                    st.info(message_text)

            # Always declare this container, even when no action is staged,
            # so the number of elements preceding the search box never
            # changes between reruns. Streamlit re-keys elements by their
            # position in the tree; letting these panels appear and
            # disappear directly in the flow shifted the search box's
            # position on every toggle, which desynced its custom
            # st_keyup iframe component (it would get stuck showing its
            # unrendered placeholder markup instead of the real widget).
            with st.container():
                if st.session_state.get("confirm_delete_pids"):
                    pids_to_delete = st.session_state.confirm_delete_pids
                    st.warning(
                        f"Delete {len(pids_to_delete)} paper(s)? This cannot be undone."
                    )
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button("Confirm", key="confirm_delete_btn"):
                            delete_succeeded = delete_selected_papers(
                                creds,
                                pids_to_delete,
                                st.session_state.index,
                                st.session_state.current_papers_id,
                                st.session_state.local_lib_dir,
                            )
                            st.session_state.confirm_delete_pids = None
                            if delete_succeeded:
                                st.rerun()
                    with cancel_col:
                        if st.button("Cancel", key="cancel_delete_btn"):
                            st.session_state.confirm_delete_pids = None
                            st.rerun()

                if st.session_state.get("confirm_generate_pids"):
                    pids_to_generate = st.session_state.confirm_generate_pids
                    st.warning(
                        f"Generate metadata for {len(pids_to_generate)} paper(s)? "
                        "Any existing metadata will be overwritten."
                    )
                    confirm_gen_col, cancel_gen_col = st.columns(2)
                    with confirm_gen_col:
                        if st.button("Confirm", key="confirm_generate_btn"):
                            generate_metadata_for_selected(
                                creds,
                                pids_to_generate,
                                st.session_state.index,
                                st.session_state.current_papers_id,
                                st.session_state.local_lib_dir,
                            )
                            st.session_state.confirm_generate_pids = None
                            st.rerun()
                    with cancel_gen_col:
                        if st.button("Cancel", key="cancel_generate_btn"):
                            st.session_state.confirm_generate_pids = None
                            st.rerun()

                if st.session_state.get("show_add_tag_pids"):
                    pids_to_tag = st.session_state.show_add_tag_pids
                    new_tags_str = st.text_input(
                        f"Add tag(s) to {len(pids_to_tag)} paper(s), comma separated",
                        key="add_tag_input",
                    )
                    add_tag_col, cancel_add_tag_col = st.columns(2)
                    with add_tag_col:
                        if st.button("Add", key="confirm_add_tag_btn"):
                            new_tags = [
                                t.strip() for t in new_tags_str.split(",") if t.strip()
                            ]
                            add_succeeded = True
                            if new_tags:
                                add_succeeded = add_tags_to_selected(
                                    creds,
                                    pids_to_tag,
                                    st.session_state.index,
                                    st.session_state.current_papers_id,
                                    st.session_state.local_lib_dir,
                                    new_tags,
                                )
                            st.session_state.show_add_tag_pids = None
                            if add_succeeded:
                                st.rerun()
                    with cancel_add_tag_col:
                        if st.button("Cancel", key="cancel_add_tag_btn"):
                            st.session_state.show_add_tag_pids = None
                            st.rerun()

                if st.session_state.get("show_remove_tag_pids"):
                    pids_to_untag = st.session_state.show_remove_tag_pids
                    tags_in_selection = sorted(
                        {
                            tag
                            for pid in pids_to_untag
                            if pid in st.session_state.index.papers
                            for tag in st.session_state.index.papers[pid].tags
                        }
                    )
                    if not tags_in_selection:
                        st.info("None of the selected papers have any tags.")
                        st.session_state.show_remove_tag_pids = None
                    else:
                        tags_to_remove = st.multiselect(
                            f"Remove tag(s) from {len(pids_to_untag)} paper(s)",
                            options=tags_in_selection,
                            key="remove_tag_select",
                        )
                        remove_tag_col, cancel_remove_tag_col = st.columns(2)
                        with remove_tag_col:
                            if st.button("Remove", key="confirm_remove_tag_btn"):
                                remove_succeeded = True
                                if tags_to_remove:
                                    remove_succeeded = remove_tags_from_selected(
                                        creds,
                                        pids_to_untag,
                                        st.session_state.index,
                                        st.session_state.current_papers_id,
                                        st.session_state.local_lib_dir,
                                        tags_to_remove,
                                    )
                                st.session_state.show_remove_tag_pids = None
                                if remove_succeeded:
                                    st.rerun()
                        with cancel_remove_tag_col:
                            if st.button("Cancel", key="cancel_remove_tag_btn"):
                                st.session_state.show_remove_tag_pids = None
                                st.rerun()

                if st.session_state.get("show_download_pids"):
                    pids_to_download = st.session_state.show_download_pids
                    zip_result = st.session_state.get("download_zip_result")
                    if zip_result is None:
                        with st.spinner("Preparing zip..."):
                            zip_result = zip_marked_pdfs(
                                creds,
                                pids_to_download,
                                st.session_state.index,
                                st.session_state.local_lib_dir,
                            )
                        st.session_state.download_zip_result = zip_result
                    if zip_result.skipped:
                        if len(zip_result.skipped) > MAX_DUPLICATE_NAMES_TO_LIST:
                            st.warning(
                                f"Couldn't include {len(zip_result.skipped)} PDFs."
                            )
                        else:
                            st.warning(
                                "Couldn't include: " + ", ".join(zip_result.skipped)
                            )
                    if zip_result.included:
                        lib_name = st.session_state.get(
                            "current_lib_name", st.session_state.current_lib_id
                        )
                        st.download_button(
                            "Download zip",
                            data=zip_result.data,
                            file_name=zip_download_filename(lib_name),
                            mime="application/zip",
                            key="confirm_download_btn",
                        )
                    else:
                        st.error("No PDFs could be included in the download.")
                    if st.button("Cancel", key="cancel_download_btn"):
                        st.session_state.show_download_pids = None
                        st.session_state.pop("download_zip_result", None)
                        st.rerun()

            search_box = st_keyup(
                "Search",
                placeholder="Search papers...",
                key=search_box_key,
                label_visibility="collapsed",
            )
            search_query = (search_box or "").lower()

            status_filter_labels = st.multiselect(
                "Status",
                options=list(STATUS_LABELS.values()) + [SIMILAR_FILTER_LABEL],
                key=status_filter_key,
                placeholder="Filter by status...",
                label_visibility="collapsed",
            )
            status_filter = [
                LABEL_TO_STATUS[label]
                for label in status_filter_labels
                if label in LABEL_TO_STATUS
            ]
            include_similar_filter = SIMILAR_FILTER_LABEL in status_filter_labels

            all_tags = get_all_tags(st.session_state.index)
            # A previously selected tag may no longer exist (its last
            # paper was deleted or retagged since the last rerun). Drop
            # it from the persisted selection before the widget reads
            # it so a stale value never lingers against the current
            # options.
            if tags_filter_key in st.session_state:
                st.session_state[tags_filter_key] = [
                    tag for tag in st.session_state[tags_filter_key] if tag in all_tags
                ]
            tags_filter = st.multiselect(
                "Tags",
                options=all_tags,
                key=tags_filter_key,
                placeholder="Filter by tags...",
                label_visibility="collapsed",
            )

            # Re-filter after the block above so a partial batch-delete
            # failure (which skips st.rerun() to keep its error visible)
            # never renders a now-deleted paper's row - clicking one would
            # otherwise select a pid missing from st.session_state.index.papers
            # and crash the main-area lookup with a KeyError.
            duplicate_pids = get_duplicate_pids(st.session_state.index)

            filtered_papers = filter_papers(
                st.session_state.index.papers,
                search_query,
                status_filter,
                tags_filter,
                duplicate_pids=duplicate_pids,
                include_similar=include_similar_filter,
            )

            st.checkbox(
                "Mark all",
                key="select_all_toggle",
                on_change=_toggle_select_all,
                args=([pid for pid, _ in filtered_papers],),
            )

            with st.container(height=400):
                if not filtered_papers:
                    if not st.session_state.index.papers:
                        st.info(
                            "Your library is empty. Upload PDFs above to get started!"
                        )
                    else:
                        st.info("No papers match your search and filters.")

                for pid, p in filtered_papers:
                    row_check, row_button = st.columns([1, 8])
                    with row_check:
                        st.checkbox(
                            "Select",
                            key=f"chk_{pid}",
                            label_visibility="collapsed",
                            on_change=_uncheck_mark_all_if_unmarked,
                            args=(pid,),
                        )
                    with row_button:
                        display_name = f"{STATUS_ICONS.get(p.status, '📄')} {p.title}"
                        if pid in duplicate_pids:
                            display_name = f"⚠️ {display_name}"
                        if pid == st.session_state.selected_paper:
                            display_name = f"**{display_name}**"
                        if st.button(
                            display_name, key=f"btn_{pid}", use_container_width=True
                        ):
                            st.session_state.selected_paper = pid
                            st.session_state.confirm_delete_pids = None
                            st.rerun()

    # Main area
    if st.session_state.selected_paper:
        pid = st.session_state.selected_paper
        if not re.match(PAPER_ID_PATTERN, pid):
            st.error("Invalid paper ID format.")
            st.stop()
        paper_info = st.session_state.index.papers[pid]

        local_paper_dir = st.session_state.local_lib_dir / pid
        local_paper_dir.mkdir(parents=True, exist_ok=True)
        local_pdf_path = local_paper_dir / PDF_FILENAME
        local_meta_path = local_paper_dir / META_FILENAME

        # Download files if missing
        pdf_available = local_pdf_path.exists()
        with st.spinner("Loading paper..."):
            if not pdf_available:
                try:
                    download_file(creds, paper_info.pdf_file_id, local_pdf_path)
                    pdf_available = True
                except Exception as e:
                    st.error(f"Failed to load PDF: {e}")

            # Always sync metadata on load to avoid stale caches across devices
            metadata_available = sync_paper_metadata(creds, paper_info, local_meta_path)

        meta = PaperMetadata(title=paper_info.title)
        if local_meta_path.exists():
            try:
                data = json.loads(local_meta_path.read_text(encoding="utf-8"))
                meta = PaperMetadata(**data)
            except ValidationError as e:
                st.warning("Metadata invalid, recovering valid fields.")
                data = json.loads(local_meta_path.read_text(encoding="utf-8"))
                invalid_fields = [
                    err.get("loc")[0] for err in e.errors() if err.get("loc")
                ]
                for field in invalid_fields:
                    if field in data:
                        del data[field]
                data["title"] = data.get("title", paper_info.title)
                try:
                    meta = PaperMetadata(**data)
                except Exception:
                    logger.exception(
                        "Failed to recover valid fields from %s", local_meta_path
                    )
                    st.warning(
                        "Could not recover this paper's saved notes/citation/"
                        "status - showing a blank metadata form instead."
                    )
                    meta = PaperMetadata(title=paper_info.title)
            except Exception as e:
                st.error(f"Could not load metadata: {e}")

        col_pdf, col_meta = st.columns([3.5, 1])
        with col_pdf:
            if pdf_available:
                local_edited_path = local_paper_dir / EDITED_PDF_FILENAME
                edited_available = local_edited_path.exists()
                if paper_info.edited_pdf_file_id and not edited_available:
                    try:
                        download_file(
                            creds, paper_info.edited_pdf_file_id, local_edited_path
                        )
                        edited_available = True
                    except Exception as e:
                        st.error(f"Failed to load edited PDF: {e}")

                view_filename = (
                    EDITED_PDF_FILENAME if edited_available else PDF_FILENAME
                )

                base_url = os.environ.get("FASTAPI_URL", DEFAULT_FASTAPI_URL)
                quoted_lib_id = urllib.parse.quote(st.session_state.current_lib_id)
                quoted_pid = urllib.parse.quote(pid)
                pdf_file_url = (
                    f"{base_url.rstrip('/')}/papers/{quoted_lib_id}/{quoted_pid}"
                    f"/{view_filename}"
                )
                if view_filename == EDITED_PDF_FILENAME:
                    # upload_file_to_folder reuses the same Drive file id on
                    # re-upload, so the URL alone wouldn't change after a
                    # further round of annotation - bust the browser's cache
                    # with the local file's mtime.
                    pdf_file_url += f"?v={local_edited_path.stat().st_mtime_ns}"

                viewer_query = urllib.parse.urlencode(
                    {
                        "file": pdf_file_url,
                        "libId": st.session_state.current_lib_id,
                        "pid": pid,
                    }
                )
                viewer_url = (
                    f"{base_url.rstrip('/')}/pdfjs/web/viewer.html?{viewer_query}"
                )

                pdf_expanded = st.session_state.get("pdf_expanded", False)
                toggle_label = "✖ Exit full screen" if pdf_expanded else "⛶ Full screen"
                # Streamlit's own header/sidebar chrome use z-index 999990/
                # 999991, so the overlay must clear that to actually cover
                # the full viewport, and the toggle button's wrapping
                # container needs an even higher z-index (via its stable
                # `st-key-*` class) so it stays reachable on top of the
                # overlay instead of being buried underneath it.
                overlay_z_index = 1_000_000
                toggle_container = st.container(key="pdf_expand_toggle_container")
                with toggle_container:
                    if st.button(toggle_label, key="pdf_expand_toggle"):
                        st.session_state.pdf_expanded = not pdf_expanded
                        st.rerun()

                # A CSS-only expand, not the Fullscreen API - PDF.js's native
                # Presentation Mode disables the annotation editor on entry,
                # which breaks the app's highlight-and-autosave workflow.
                wrapper_class = (
                    "pdf-viewer-expanded" if pdf_expanded else "pdf-viewer-normal"
                )
                # Parked in the lower-right corner while expanded so it
                # doesn't overlap the PDF.js toolbar (top) or sidebar tools.
                toggle_float_css = (
                    ".st-key-pdf_expand_toggle_container{position:fixed !"
                    "important;bottom:1.5rem;right:1.5rem;"
                    f"z-index:{overlay_z_index + 1} !important;"
                    "width:fit-content !important;}"
                    if pdf_expanded
                    else ""
                )
                pdf_display = (
                    "<style>.pdf-viewer-expanded{position:fixed;inset:0;"
                    "width:100vw;height:100vh;"
                    f"z-index:{overlay_z_index};background:#fff;}}"
                    ".pdf-viewer-expanded iframe{width:100%;height:100%;}"
                    f"{toggle_float_css}"
                    "</style>"
                    f'<div class="{wrapper_class}">'
                    f'<iframe src="{html.escape(viewer_url)}" width="100%" '
                    'height="750" style="border:none;" allow="fullscreen" '
                    "allowfullscreen></iframe>"
                    "</div>"
                )
                st.markdown(pdf_display, unsafe_allow_html=True)

                st.caption(
                    "Use the toolbar's highlight tool to annotate the PDF - "
                    "your edits save to Drive automatically as you work. "
                    "Use Expand to fill the browser window while keeping "
                    "annotation tools active."
                )
            else:
                st.warning("PDF could not be loaded from Drive.")

        with col_meta:
            st.subheader("Metadata")
            if not metadata_available:
                st.warning(
                    "Could not load the latest metadata from Drive. Editing is "
                    "disabled to avoid overwriting your saved data."
                )

            hf_token_configured = is_hf_token_configured()
            if st.button(
                "✨ Generate metadata",
                key=f"generate_btn_{pid}",
                disabled=not pdf_available or not hf_token_configured,
                help=(
                    HF_TOKEN_MISSING_HELP
                    if not hf_token_configured
                    else "PDF must be downloaded to generate metadata"
                    if not pdf_available
                    else GENERATE_METADATA_HELP
                ),
            ):
                has_unsaved_edits = (
                    st.session_state.get(f"title_{pid}", meta.title) != meta.title
                    or st.session_state.get(f"abstract_{pid}", meta.abstract)
                    != meta.abstract
                    or st.session_state.get(f"tags_{pid}", ", ".join(meta.tags))
                    != ", ".join(meta.tags)
                )
                if meta.abstract or meta.tags or has_unsaved_edits:
                    st.session_state[f"confirm_regenerate_{pid}"] = True
                else:
                    with st.spinner("Generating metadata with Hugging Face..."):
                        if generate_metadata_for_paper(pid, local_pdf_path):
                            st.rerun()

            if st.session_state.get(f"confirm_regenerate_{pid}"):
                st.warning(
                    "This paper already has generated metadata or unsaved "
                    "edits. Regenerate and overwrite them?"
                )
                regen_col, cancel_regen_col = st.columns(2)
                with regen_col:
                    if st.button("Regenerate", key=f"confirm_regenerate_btn_{pid}"):
                        st.session_state.pop(f"confirm_regenerate_{pid}", None)
                        with st.spinner("Generating metadata with Hugging Face..."):
                            if generate_metadata_for_paper(pid, local_pdf_path):
                                st.rerun()
                with cancel_regen_col:
                    if st.button("Cancel", key=f"cancel_regenerate_btn_{pid}"):
                        st.session_state.pop(f"confirm_regenerate_{pid}", None)
                        st.rerun()

            draft = st.session_state.get(f"generated_{pid}", {})
            dupe_embedding = draft.get("embedding") or paper_info.embedding
            if dupe_embedding:
                for _, dupe_title, dupe_score in find_similar_papers(
                    dupe_embedding, st.session_state.index, exclude_pid=pid
                ):
                    st.warning(f"Similar to '{dupe_title}' — {dupe_score:.0%} match")

            with st.form(key=f"meta_form_{pid}"):
                new_title = st.text_input(
                    "Title", value=draft.get("title", meta.title), key=f"title_{pid}"
                )
                new_abstract = st.text_area(
                    "Abstract / TL;DR",
                    value=draft.get("abstract", meta.abstract),
                    height=100,
                    key=f"abstract_{pid}",
                )
                tags_str = st.text_input(
                    "Tags (comma separated)",
                    value=", ".join(draft.get("tags", meta.tags)),
                    key=f"tags_{pid}",
                )
                status_label = st.selectbox(
                    "Status",
                    options=list(STATUS_LABELS.values()),
                    index=list(STATUS_LABELS.keys()).index(meta.status),
                    key=f"status_{pid}",
                )
                status = LABEL_TO_STATUS.get(status_label, meta.status)
                citation = st.text_input(
                    "Citation", value=meta.citation, key=f"citation_{pid}"
                )
                notes = st.text_area(
                    "Notes", value=meta.notes, height=200, key=f"notes_{pid}"
                )

                if st.form_submit_button(
                    "Save Changes", disabled=not metadata_available
                ):
                    meta = meta.model_copy(
                        update={
                            "title": strip_pdf_suffix(new_title or ""),
                            "abstract": new_abstract or "",
                            "tags": [
                                t.strip() for t in tags_str.split(",") if t.strip()
                            ],
                            "status": cast(
                                Literal["Unread", "Reading", "Read", "TODO"], status
                            ),
                            "citation": citation,
                            "notes": notes,
                            "embedding": draft.get("embedding", meta.embedding),
                        }
                    )

                    local_meta_path.write_text(
                        meta.model_dump_json(indent=2), encoding="utf-8"
                    )
                    with st.spinner("Saving metadata to Drive..."):
                        upload_file_to_folder(
                            creds,
                            paper_info.folder_id,
                            local_meta_path,
                            META_FILENAME,
                            JSON_MIME_TYPE,
                        )

                        paper_info = paper_info.model_copy(
                            update={
                                "title": meta.title,
                                "tags": meta.tags,
                                "status": meta.status,
                                "embedding": meta.embedding,
                            }
                        )
                        st.session_state.index.papers[pid] = paper_info
                        upload_library_index(
                            creds,
                            st.session_state.current_papers_id,
                            st.session_state.index,
                        )

                    st.session_state.pop(f"generated_{pid}", None)
                    st.success("Metadata saved!")
                    st.rerun()
    else:
        st.info("Select a paper from the sidebar to view it.")


if __name__ == "__main__":
    main()
