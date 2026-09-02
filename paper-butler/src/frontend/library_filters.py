"""Pure query/filter helpers over a LibraryIndex, for sidebar display."""

import math
import re
from typing import Iterable, Sequence

import streamlit as st

from backend.huggingface_client import DEFAULT_DUPLICATE_THRESHOLD
from backend.models import LibraryIndex, PaperIndexEntry
from frontend.constants import PAPER_ID_PATTERN


def get_all_tags(index: LibraryIndex) -> list[str]:
    """Returns every distinct tag used across the library, sorted alphabetically.

    Args:
        index: The library index to scan.

    Returns:
        list[str]: The sorted, deduplicated tags used by any paper in `index`.
    """
    return sorted({tag for p in index.papers.values() for tag in p.tags})


def get_duplicate_pids(index: LibraryIndex) -> set[str]:
    """Finds every paper whose embedding matches another paper in the index.

    Computed fresh from the persisted embeddings already in `index` (rather
    than from the ephemeral `dupes_{pid}` session-state key that's only
    populated right after generation), so the result is available for any
    paper regardless of when its embedding was generated or whether the
    user has navigated away and back.

    The underlying pairwise comparison is O(N^2) in the number of papers,
    but this is called on every Streamlit rerun (i.e. on every click or
    keystroke anywhere in the app), so the result is cached in
    `st.session_state` under a signature of every paper's embedding
    content. `st.session_state.index` is typically the same object mutated
    in place across reruns, so caching by object identity wouldn't detect
    embedding changes - the signature is recomputed each call (an O(N)
    operation) and only triggers the expensive O(N^2) scan when it
    actually differs from the last cached signature.

    Args:
        index: The library index to scan.

    Returns:
        set[str]: The paper IDs with at least one similar-embedding match
        elsewhere in the index, per `find_similar_papers`'s default
        threshold.
    """
    signature = frozenset(
        (pid, tuple(entry.embedding))
        for pid, entry in index.papers.items()
        if entry.embedding
    )
    cached = st.session_state.get("_duplicate_pids_cache")
    if cached is not None and cached[0] == signature:
        return cached[1]

    valid_papers = [
        (pid, entry.embedding) for pid, entry in index.papers.items() if entry.embedding
    ]
    n = len(valid_papers)

    # Pre-calculate normalized embeddings to avoid duplicate work in O(N^2) loops
    normalized_embs = []
    for _, emb in valid_papers:
        norm = math.hypot(*emb)
        if norm == 0.0:
            normalized_embs.append(None)
        else:
            normalized_embs.append([x / norm for x in emb])

    result = set()
    for i in range(n):
        emb1 = normalized_embs[i]
        if emb1 is None:
            continue

        for j in range(i + 1, n):
            emb2 = normalized_embs[j]
            if emb2 is None or len(emb1) != len(emb2):
                continue

            score = math.sumprod(emb1, emb2)
            if score >= DEFAULT_DUPLICATE_THRESHOLD:
                result.add(valid_papers[i][0])
                result.add(valid_papers[j][0])

    st.session_state["_duplicate_pids_cache"] = (signature, result)
    return result


def get_missing_metadata_pids(index: LibraryIndex) -> set[str]:
    """Finds every paper with no generated tags or embedding yet.

    Args:
        index: The library index to scan.

    Returns:
        set[str]: The paper IDs that have never had metadata generated.
    """
    return {
        pid
        for pid, entry in index.papers.items()
        if not entry.tags and not entry.embedding
    }


def filter_papers(
    papers: dict[str, PaperIndexEntry],
    search_query: str,
    status_filter: Sequence[str],
    tags_filter: Sequence[str],
    duplicate_pids: Iterable[str] = (),
    include_similar: bool = False,
) -> list[tuple[str, PaperIndexEntry]]:
    """Filters and sorts the library's papers for sidebar display.

    Args:
        papers: Mapping of paper ID to its index entry, as stored in
            LibraryIndex.papers.
        search_query: Lowercased search text; a paper matches if its title
            contains this text.
        status_filter: Statuses to restrict results to. Combined with
            `include_similar` via OR: empty and `include_similar=False`
            together mean no status filtering.
        tags_filter: Tags to restrict results to; a paper matches if it has
            at least one of these tags. Empty means no tag filtering.
        duplicate_pids: Paper IDs currently flagged as having a similar-
            embedding match elsewhere in the library (from
            `get_duplicate_pids`), consulted only when `include_similar` is
            True.
        include_similar: If True, a paper also matches when its ID is in
            `duplicate_pids`, regardless of `status_filter`. Lets a "⚠️
            Similar" pseudo-option sit in the same multiselect as the real
            statuses without being one itself - a paper's reading status
            and its similar-embedding flag are independent.

    Returns:
        list[tuple[str, PaperIndexEntry]]: Matching (paper_id, entry) pairs,
        sorted by title. Entries whose key isn't a 32-character hex paper ID
        (e.g. a legacy/malformed index entry) are always skipped.
    """
    duplicate_pid_set = set(duplicate_pids)
    matches: list[tuple[str, PaperIndexEntry]] = []
    for pid, p in papers.items():
        if not re.match(PAPER_ID_PATTERN, pid):
            continue
        if search_query not in p.title.lower():
            continue
        if status_filter or include_similar:
            status_matches = p.status in status_filter
            similar_matches = include_similar and pid in duplicate_pid_set
            if not (status_matches or similar_matches):
                continue
        if tags_filter and not any(tag in p.tags for tag in tags_filter):
            continue
        matches.append((pid, p))
    return sorted(matches, key=lambda x: x[1].title)
