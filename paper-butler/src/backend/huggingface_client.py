"""Hugging Face Inference Providers client for on-demand paper metadata generation."""

import json
import operator
import os
import re
import time
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, Field
from pypdf import PdfReader

from backend.models import LibraryIndex

T = TypeVar("T")

DEFAULT_GENERATION_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
MAX_EXTRACTED_CHARS: int = 6000
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: float = 2.0
DEFAULT_DUPLICATE_THRESHOLD: float = 0.90
MAX_TAGS: int = 8
REFERENCES_SEARCH_OFFSET: int = 500
"""Character offset to start searching for a References/Bibliography heading
from, skipping a paper's front matter where the term is unlikely to appear
as a section heading."""


class HFTokenMissingError(RuntimeError):
    """Raised when no Hugging Face API token is available to make a request.

    The token must have the "Make calls to Inference Providers" scope.
    """


class GeneratedMetadata(BaseModel):
    """Hugging Face-generated metadata for a paper, staged for user review.

    Attributes:
        title: Suggested paper title.
        abstract: Suggested abstract/TL;DR.
        tags: Suggested tag strings.
    """

    model_config = {"frozen": True}

    title: str
    abstract: str
    tags: List[str] = Field(default_factory=list)


def _clean_extracted_text(text: str) -> str:
    """Removes common PDF-extraction noise before it's sent to the model.

    Rejoins hyphenated line-wraps, drops any trailing References/
    Bibliography section (rarely useful for title/abstract/tag generation
    and often the single largest consumer of the truncation budget), and
    collapses excess whitespace - all to reclaim more of `max_chars` for
    actual paper content instead of layout artifacts.

    Args:
        text: Raw text joined from a PDF's pages.

    Returns:
        The cleaned text, stripped of leading/trailing whitespace.
    """
    text = re.sub(r"-\n(?=\w)", "", text)
    match = re.search(
        r"\n\s*(references|bibliography)\s*\n",
        text[REFERENCES_SEARCH_OFFSET:],
        re.IGNORECASE,
    )
    if match:
        text = text[: REFERENCES_SEARCH_OFFSET + match.start()]
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path, max_chars: int = MAX_EXTRACTED_CHARS) -> str:
    """Extracts text from a PDF file for use as model input.

    Args:
        pdf_path: Path to the local PDF file.
        max_chars: Maximum number of characters to return, truncating any
            excess so requests stay within model context limits.

    Returns:
        The concatenated text of every page read (see below), cleaned of
        common extraction noise (see `_clean_extracted_text`) and truncated
        to `max_chars`. Returns an empty string if the PDF has no
        extractable text (e.g. a scanned/image-only document) rather than
        raising.

    Raises:
        ValueError: If the PDF file cannot be read/parsed (e.g. corrupt file).
    """
    try:
        reader = PdfReader(str(pdf_path))
        chunks: List[str] = []
        total_chars = 0
        # Stop once enough raw text is collected to fill max_chars, rather
        # than extracting every page of a long document just to discard
        # most of it below. Cleaning (e.g. dropping a trailing References/
        # Bibliography section) can shrink the text further, so the final
        # cleaned+truncated output may end up shorter than max_chars.
        for page in reader.pages:
            page_text = page.extract_text() or ""
            chunks.append(page_text)
            total_chars += len(page_text)
            if total_chars >= max_chars:
                break
        text = "\n".join(chunks)
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}") from e
    return _clean_extracted_text(text)[:max_chars]


def get_inference_client(token: Optional[str] = None) -> InferenceClient:
    """Builds a Hugging Face InferenceClient, resolving the API token.

    Args:
        token: An explicit Hugging Face API token. If not provided, falls
            back to the `HF_TOKEN` environment variable.

    Returns:
        A configured InferenceClient.

    Raises:
        HFTokenMissingError: If neither `token` nor the `HF_TOKEN`
            environment variable is set.
    """
    resolved = token or os.environ.get("HF_TOKEN")
    if not resolved:
        raise HFTokenMissingError(
            "No Hugging Face API token found. Set the HF_TOKEN environment "
            'variable to a fine-grained token with the "Make calls to '
            'Inference Providers" scope.'
        )
    return InferenceClient(token=resolved)


def is_hf_token_configured() -> bool:
    """Checks whether an HF_TOKEN is available in the environment.

    Returns:
        True if the HF_TOKEN environment variable is set to a non-empty
        value, False otherwise.
    """
    return bool(os.environ.get("HF_TOKEN"))


def _call_with_retry(
    fn: Callable[[], T],
    max_retries: int = MAX_RETRIES,
    delay_seconds: float = RETRY_DELAY_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """Retries a Hugging Face API call on transient 503 "model loading" errors.

    Args:
        fn: A zero-argument callable making the API request.
        max_retries: Maximum number of attempts before giving up.
        delay_seconds: Delay passed to `sleep_fn` between retries.
        sleep_fn: Called between retries; injected so tests never sleep for
            real.

    Returns:
        The return value of `fn` on success.

    Raises:
        Exception: Re-raises the last exception if all attempts fail, or
            immediately for any non-503 error.
    """
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except HfHubHTTPError as e:
            status_code = getattr(e.response, "status_code", None)
            if status_code != 503:
                raise
            last_error = e
            if attempt < max_retries - 1:
                sleep_fn(delay_seconds)
    assert last_error is not None
    raise last_error


def _build_combined_prompt_messages(
    pdf_text: str,
    existing_tags: Sequence[str] = (),
) -> List[dict[str, str]]:
    """Builds the chat messages for the single metadata-generation call.

    Asks the model to produce the title, abstract, and tags together as one
    JSON object, instead of one call per field, to cut the number of
    Hugging Face requests per paper.

    Args:
        pdf_text: Extracted paper text to generate from.
        existing_tags: Tags already used elsewhere in the library, to bias
            generation toward reusing them instead of inventing
            near-duplicate new tags.

    Returns:
        A list of `{"role": ..., "content": ...}` messages suitable for
        `InferenceClient.chat_completion`.
    """
    instruction = (
        "You are a scientific paper assistant. Read the paper text below "
        "and respond with ONLY a single JSON object - no markdown code "
        "fences, no preamble, no commentary - with exactly these keys: "
        '"title" (a short, accurate title, no quotes), "abstract" (a '
        'concise 2-4 sentence summary/TL;DR), and "tags" (an array of up '
        "to 8 short topical tag strings)."
    )
    if existing_tags:
        instruction += (
            " Prefer reusing tags from this existing set when they fit: "
            + ", ".join(existing_tags)
            + ". Only introduce a new tag if none of these apply."
        )
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": pdf_text},
    ]


def _extract_json_object(content: str) -> str:
    """Isolates a JSON object from a chat model's raw response text.

    Strips a ```` ```json ... ``` ```` code fence if present, then narrows
    to the substring between the first `{` and the last `}` - some models
    add stray preamble/postamble around the JSON despite instructions not
    to.

    Args:
        content: The raw response content.

    Returns:
        The best-guess JSON substring, or the stripped input unchanged if
        no `{`/`}` pair is found (so the caller's `json.loads` raises a
        clear error instead of silently misbehaving).
    """
    stripped = content.strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _parse_generated_metadata(content: str) -> GeneratedMetadata:
    """Parses a chat model's raw response into validated generated metadata.

    Args:
        content: The raw response content from the generation call.

    Returns:
        A GeneratedMetadata with the parsed title, abstract, and up to
        `MAX_TAGS` deduplicated (case-insensitive) tags.

    Raises:
        ValueError: If the response is not valid JSON, or is valid JSON
            that isn't a JSON object (e.g. a bare string or array).
    """
    try:
        payload = json.loads(_extract_json_object(content))
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Hugging Face returned a non-JSON response: {content!r}"
        ) from e
    if not isinstance(payload, dict):
        raise ValueError(
            f"Hugging Face returned a JSON value that isn't an object: {content!r}"
        )

    title = str(payload.get("title", "")).strip()
    abstract = str(payload.get("abstract", "")).strip()
    raw_tags: list[object] = payload.get("tags", [])
    if not isinstance(raw_tags, list):
        raw_tags = []

    seen: set[str] = set()
    tags: List[str] = []
    for tag in raw_tags:
        stripped = str(tag).strip()
        if stripped and stripped.lower() not in seen:
            seen.add(stripped.lower())
            tags.append(stripped)
        if len(tags) >= MAX_TAGS:
            break

    return GeneratedMetadata(title=title, abstract=abstract, tags=tags)


def generate_paper_metadata(
    pdf_text: str,
    model: str = DEFAULT_GENERATION_MODEL,
    client: Optional[InferenceClient] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    existing_tags: Sequence[str] = (),
) -> GeneratedMetadata:
    """Generates a title, abstract, and tags for a paper via Hugging Face.

    Args:
        pdf_text: Extracted paper text to generate from.
        model: The chat/instruct model to call.
        client: An existing InferenceClient to reuse. If not provided, one
            is created via `get_inference_client()`.
        sleep_fn: Passed through to the retry helper for each subtask call.
        existing_tags: Tags already used elsewhere in the library, passed to
            the tag-generation subtask so it prefers reusing them over
            inventing near-duplicate new tags.

    Returns:
        A GeneratedMetadata with the suggested title, abstract, and tags.

    Raises:
        HFTokenMissingError: If no client is given and no HF token is set.
        ValueError: If the model's response is not valid JSON, or is valid
            JSON that isn't a JSON object (e.g. a bare string or array).
        Exception: Propagates any Hugging Face API error surviving retries.
    """
    active_client = client or get_inference_client()
    response = _call_with_retry(
        lambda: active_client.chat_completion(
            model=model,
            messages=_build_combined_prompt_messages(pdf_text, existing_tags),
            temperature=0,
        ),
        sleep_fn=sleep_fn,
    )
    content = (response.choices[0].message.content or "").strip()
    return _parse_generated_metadata(content)


def embed_text(
    text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
    client: Optional[InferenceClient] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> List[float]:
    """Computes a sentence embedding for the given text via Hugging Face.

    Args:
        text: The text to embed.
        model: The embedding model to call.
        client: An existing InferenceClient to reuse. If not provided, one
            is created via `get_inference_client()`.
        sleep_fn: Passed through to the retry helper.

    Returns:
        A flat list of `EMBEDDING_DIM` floats.

    Raises:
        HFTokenMissingError: If no client is given and no HF token is set.
        ValueError: If the response cannot be normalized to a vector of the
            expected dimension.
        Exception: Propagates any Hugging Face API error surviving retries.
    """
    active_client = client or get_inference_client()
    result = _call_with_retry(
        lambda: active_client.feature_extraction(text, model=model),
        sleep_fn=sleep_fn,
    )
    vector = result.tolist() if hasattr(result, "tolist") else list(result)

    # Some feature-extraction endpoints wrap the per-token (2D) response in
    # an extra batch dimension (3D: a single-element list containing the 2D
    # matrix) - unwrap that before pooling, or zip(*rows) below would zip
    # over the single 2D row instead of over tokens.
    if vector and isinstance(vector[0], list) and isinstance(vector[0][0], list):
        vector = vector[0]

    # Mean-pool a per-token (2D) response down to a single sentence vector,
    # in pure Python so this module doesn't need its own numpy dependency.
    while vector and isinstance(vector[0], list):
        rows = vector
        vector = [sum(col) / len(rows) for col in zip(*rows)]

    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Expected a {EMBEDDING_DIM}-dim embedding, got {len(vector)} "
            f"from model {model!r}."
        )
    return vector


def cosine_similarity(
    a: Sequence[float], b: Sequence[float], norm_a: Optional[float] = None
) -> float:
    """Computes the cosine similarity between two vectors.

    Args:
        a: The first vector.
        b: The second vector.
        norm_a: `a`'s pre-computed Euclidean norm, if the caller already has
            it (e.g. from comparing `a` against many vectors in a loop).
            Computed from `a` when omitted.

    Returns:
        A similarity score in [-1, 1], or 0.0 if either vector is empty or
        has zero norm (the expected "no embedding yet" case).

    Raises:
        ValueError: If `a` and `b` have different lengths.
    """
    if len(a) != len(b):
        raise ValueError("vectors must be the same length")
    if not a or not b:
        return 0.0
    dot = sum(map(operator.mul, a, b))
    if norm_a is None:
        norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_similar_papers(
    embedding: Sequence[float],
    index: LibraryIndex,
    exclude_pid: Optional[str] = None,
    threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
) -> List[Tuple[str, str, float]]:
    """Finds papers in the library whose embedding is similar to `embedding`.

    Args:
        embedding: The embedding to compare against every paper in `index`.
        index: The library index to scan.
        exclude_pid: A paper ID to skip (typically the paper being generated
            for, so it's never flagged as its own duplicate).
        threshold: Minimum cosine similarity score to be considered a match.

    Returns:
        A list of `(paper_id, title, score)` tuples for every paper at or
        above `threshold`, sorted by score descending. Papers with no
        stored embedding yet, or whose embedding has a different dimension
        than `embedding` (e.g. a legacy/corrupted entry), are excluded. If
        `embedding` is empty or has zero norm, every score against it would
        be 0.0; that case short-circuits to `[]` immediately whenever
        `threshold > 0.0` (nothing could match), and otherwise still scores
        each paper normally so a non-positive `threshold` behaves exactly
        as if the loop had run.
    """
    # Pre-calculate the query embedding's norm once so cosine_similarity()
    # doesn't recompute it for every paper in the library.
    norm_query = sum(x * x for x in embedding) ** 0.5 if embedding else 0.0
    if norm_query == 0.0 and threshold > 0.0:
        return []

    matches = []
    for pid, entry in index.papers.items():
        if pid == exclude_pid or not entry.embedding:
            continue
        if len(embedding) != len(entry.embedding):
            continue
        score = cosine_similarity(embedding, entry.embedding, norm_a=norm_query)
        if score >= threshold:
            matches.append((pid, entry.title, score))
    matches.sort(key=lambda m: m[2], reverse=True)
    return matches
