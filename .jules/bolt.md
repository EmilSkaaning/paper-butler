## 2024-05-18 - Optimized Duplicate Detection bottleneck
**Learning:** The `get_duplicate_pids` sidebar filter function had an $O(N^2)$ bottleneck because it repeatedly called `find_similar_papers`, performing redundant vector normalizations for the cosine similarity math.
**Action:** When implementing mathematical comparisons that iterate over pairs in a list, compute invariants (like vector norms) outside the loops. We optimized `get_duplicate_pids` with a double loop (`i`, `j = i + 1`) and pre-calculated norms for a ~5x speedup.
## 2024-05-19 - Duplicate API call overhead
**Learning:** `find_similar_papers` had an O(N) function call overhead bottleneck due to calling `cosine_similarity` for every document comparison in the loop. Furthermore, the query norm was being redundantly calculated inside `cosine_similarity` for each individual comparison.
**Action:** Inline math calculations when iterating over collections and pre-compute constants. We moved the `norm_query` logic to the start of `find_similar_papers` and inlined the vector dot product loop, yielding an approximately ~30% improvement in matching speed.
## 2024-05-20 - Cache key signature bottleneck with large lists (correction)
**Learning:** An earlier version of this note recommended replacing content-based cache
signatures with `frozenset((id_field, id(array_field)))` for speed. That is unsafe: CPython
reuses the memory address of a freed list for a new, unrelated list, so `id()` is not a
content fingerprint — it produces false cache hits (stale, silently wrong results) once an old
embedding list is garbage-collected. `frozen=True` on the Pydantic model only blocks attribute
*reassignment*; it does not stop the underlying list from being mutated or freed and reused.
**Action:** Keep cache signatures content-based. To avoid the O(N log N) sort cost while
staying correct, drop `sorted()` and hash a `frozenset` of `(pid, tuple(embedding))` pairs
directly — order doesn't matter for equality, so the sort was pure overhead, not the `tuple()`
conversion. Never use object identity (`id()`) as a proxy for mutable-value equality in a cache
key.
## 2024-05-21 - C-level vector operations speed up mathematical operations over large lists
**Learning:** Python generator expressions like `sum(x * y for x, y in zip(a, b))` and `sum(x * x for x in a)` are inefficient for dot products or L2 norms on large lists (like 384-dimensional embeddings) due to Python-level loops.
**Action:** Use `sum(map(operator.mul, a, b))` and `sum(map(operator.mul, a, a))` instead. This pushes iteration and multiplication to the C-level, yielding a measurable speedup (e.g., ~1.3-1.6x faster).
