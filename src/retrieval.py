"""BM25 retrieval over the persisted chunk index."""

from collections import Counter, OrderedDict
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .embedding import DEFAULT_EMBEDDING_MODEL, EmbeddingIndex
from .indexing import Index, IndexError_
from .models import MinimalSource
from .tokenization import tokenize

BM25_K1 = 1.5
BM25_B = 0.75
MAX_CHUNKS_PER_FILE = 3
QUERY_CACHE_SIZE = 512
HYBRID_ALPHA = 0.9
HYBRID_POOL = 100

CacheKey = Tuple[str, int, int]


class Retriever:
    """Ranks corpus chunks against a natural-language query.

    Attributes:
        index: The inverted index the retriever queries.
    """

    def __init__(self, index: Index, cache_size: int = QUERY_CACHE_SIZE) -> None:
        """Precompute the collection statistics BM25 needs.

        Args:
            index: The inverted index to query.
            cache_size: Number of query results kept in the LRU cache; zero
                disables caching.
        """
        self.index = index
        self._cache_size = max(cache_size, 0)
        self._cache: "OrderedDict[CacheKey, List[MinimalSource]]" = OrderedDict()
        lengths = index.chunk_lens.astype(np.float32)
        self._average_length = float(lengths.mean()) if lengths.size else 1.0
        self._normalized_lengths = lengths / (self._average_length or 1.0)
        document_freqs = np.diff(index.indptr).astype(np.float64)
        total = float(index.chunk_count)
        self._idf = np.log(
            1.0 + (total - document_freqs + 0.5) / (document_freqs + 0.5)
        ).astype(np.float32)

    def score(self, query: str) -> Optional[NDArray[np.float32]]:
        """Score every chunk against ``query``.

        Args:
            query: The natural-language query.

        Returns:
            One BM25 score per chunk, or None when no query term is known to
            the index.
        """
        counts = Counter(tokenize(query))
        if not counts:
            return None
        scores = np.zeros(self.index.chunk_count, dtype=np.float32)
        matched = False
        for term, query_freq in counts.items():
            term_id = self.index.vocabulary.get(term)
            if term_id is None:
                continue
            start = int(self.index.indptr[term_id])
            end = int(self.index.indptr[term_id + 1])
            if start == end:
                continue
            chunk_ids = self.index.doc_ids[start:end]
            freqs = self.index.term_freqs[start:end]
            denominator = freqs + BM25_K1 * (
                1.0 - BM25_B + BM25_B * self._normalized_lengths[chunk_ids]
            )
            weight = float(self._idf[term_id]) * float(query_freq) * (BM25_K1 + 1.0)
            scores[chunk_ids] += weight * freqs / denominator
            matched = True
        return scores if matched else None

    def search(
        self, query: str, k: int, max_per_file: int = MAX_CHUNKS_PER_FILE
    ) -> List[MinimalSource]:
        """Return the ``k`` best chunks for ``query``.

        Chunks of a same file are capped so that one strongly matching file
        cannot monopolise the whole result list. Results are memoised, which
        matters for the API where the same question is often asked twice.

        Args:
            query: The natural-language query.
            k: Maximum number of sources to return.
            max_per_file: Maximum number of chunks kept per file.

        Returns:
            The ranked source locations, best first; empty when nothing
            matches.
        """
        if k <= 0:
            return []
        key = (query, k, max_per_file)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return list(cached)
        return list(self._remember(key, self._search(query, k, max_per_file)))

    def _remember(
        self, key: CacheKey, sources: List[MinimalSource]
    ) -> List[MinimalSource]:
        """Store a result in the LRU cache, evicting the oldest entry.

        Args:
            key: The query, cutoff and per-file cap the result was built with.
            sources: The result to store.

        Returns:
            The stored result.
        """
        if self._cache_size:
            self._cache[key] = sources
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return sources

    def _search(
        self, query: str, k: int, max_per_file: int
    ) -> List[MinimalSource]:
        """Rank the corpus against ``query``, bypassing the cache.

        Args:
            query: The natural-language query.
            k: Maximum number of sources to return.
            max_per_file: Maximum number of chunks kept per file.

        Returns:
            The ranked source locations, best first.
        """
        scores = self.score(query)
        if scores is None:
            return []
        for candidate in self._ranked_candidates(scores, k):
            selected = self._select(candidate, k, max_per_file)
            if len(selected) >= k or candidate.size >= scores.size:
                return selected
        return self._select(np.argsort(-scores, kind="stable"), k, max_per_file)

    def _ranked_candidates(
        self, scores: NDArray[np.float32], k: int
    ) -> List[NDArray[np.int64]]:
        """Yield growing pools of best-scoring chunk identifiers.

        Args:
            scores: Per-chunk scores.
            k: Number of sources ultimately requested.

        Returns:
            Candidate pools ordered by score, from narrow to wide.
        """
        pools = []
        for factor in (4, 32):
            size = min(scores.size, max(k * factor, k))
            partition = np.argpartition(-scores, size - 1)[:size]
            pools.append(partition[np.argsort(-scores[partition], kind="stable")])
            if size == scores.size:
                break
        return pools

    def _select(
        self, candidates: NDArray[np.int64], k: int, max_per_file: int
    ) -> List[MinimalSource]:
        """Turn ranked chunk identifiers into sources under a per-file cap.

        Args:
            candidates: Chunk identifiers ordered by decreasing score.
            k: Maximum number of sources to return.
            max_per_file: Maximum number of chunks kept per file.

        Returns:
            The selected source locations, best first.
        """
        selected: List[MinimalSource] = []
        seen: Counter[int] = Counter()
        for chunk_id in candidates:
            file_id = int(self.index.chunk_files[chunk_id])
            if max_per_file > 0 and seen[file_id] >= max_per_file:
                continue
            seen[file_id] += 1
            selected.append(self.source_of(int(chunk_id)))
            if len(selected) >= k:
                break
        return selected

    def source_of(self, chunk_id: int) -> MinimalSource:
        """Describe a chunk as a source location.

        Args:
            chunk_id: Identifier of the chunk.

        Returns:
            The chunk's file path and character span.
        """
        return MinimalSource(
            file_path=self.index.file_paths[int(self.index.chunk_files[chunk_id])],
            first_character_index=int(self.index.chunk_first[chunk_id]),
            last_character_index=int(self.index.chunk_last[chunk_id]),
        )


class HybridRetriever(Retriever):
    """Reranks the BM25 shortlist with dense cosine similarity.

    Lexical matching decides which chunks are considered at all; the semantic
    score only reorders that shortlist. Letting the dense score select
    candidates outright measurably hurts recall on this corpus, where queries
    quote identifiers verbatim.
    """

    def __init__(
        self,
        index: Index,
        embeddings: EmbeddingIndex,
        alpha: float = HYBRID_ALPHA,
        pool: int = HYBRID_POOL,
        cache_size: int = QUERY_CACHE_SIZE,
    ) -> None:
        """Pair a lexical index with its chunk vectors.

        Args:
            index: The inverted index to query.
            embeddings: Chunk vectors covering ``index``.
            alpha: Weight of the lexical score; the semantic score gets the
                remainder.
            pool: Number of BM25 candidates handed to the reranker.
            cache_size: Number of query results kept in the LRU cache.

        Raises:
            ValueError: When the vectors do not cover the index.
        """
        super().__init__(index, cache_size)
        if embeddings.chunk_count != index.chunk_count:
            raise ValueError(
                f"embeddings cover {embeddings.chunk_count} chunks but the "
                f"index holds {index.chunk_count}; rebuild them together"
            )
        self.embeddings = embeddings
        self.alpha = min(max(alpha, 0.0), 1.0)
        self.pool = max(pool, 1)

    def _search(self, query: str, k: int, max_per_file: int) -> List[MinimalSource]:
        """Rank the BM25 shortlist by the fused score.

        Args:
            query: The natural-language query.
            k: Maximum number of sources to return.
            max_per_file: Maximum number of chunks kept per file.

        Returns:
            The ranked source locations, best first.
        """
        scores = self.score(query)
        if scores is None:
            return []
        size = min(scores.size, max(self.pool, k))
        candidates = np.argpartition(-scores, size - 1)[:size]
        lexical = _rescale(scores[candidates])
        dense = self.embeddings.vectors[candidates] @ self.embeddings.encode_query(
            query
        )
        fused = self.alpha * lexical + (1.0 - self.alpha) * dense
        ranked = candidates[np.argsort(-fused, kind="stable")]
        return self._select(ranked, k, max_per_file)


def _rescale(scores: NDArray[np.float32]) -> NDArray[np.float32]:
    """Map scores onto ``[0, 1]`` so they can be mixed with cosines.

    Args:
        scores: The raw BM25 scores of the shortlist.

    Returns:
        The rescaled scores; all ones when the shortlist is flat.
    """
    low, high = float(scores.min()), float(scores.max())
    if high <= low:
        return np.ones_like(scores)
    return (scores - low) / (high - low)


def load_retriever(
    index_directory: str,
    hybrid: bool = False,
    alpha: float = HYBRID_ALPHA,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[Retriever, Index]:
    """Load an index from disk and wrap it in a retriever.

    Args:
        index_directory: Directory holding the persisted index.
        hybrid: Whether to rerank with chunk embeddings when they exist.
        alpha: Weight of the lexical score in the fusion.
        model_name: Identifier of the model queries are encoded with.

    Returns:
        The retriever and the index it queries.

    Raises:
        IndexError_: When hybrid retrieval is asked for but no usable
            embeddings sit next to the index.
    """
    index = Index.load(index_directory)
    if not hybrid:
        return Retriever(index), index
    embeddings = EmbeddingIndex.load(index_directory, model_name)
    if embeddings is None:
        raise IndexError_(
            f"no embeddings in '{index_directory}'; "
            "run the index command with --embeddings first"
        )
    try:
        return HybridRetriever(index, embeddings, alpha), index
    except ValueError as error:
        raise IndexError_(str(error)) from error
