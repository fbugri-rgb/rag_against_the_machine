"""Corpus ingestion and persistence of the inverted index.

The index is stored as a term-major CSR structure: ``indptr`` delimits, for
every term of the vocabulary, the slice of ``doc_ids``/``term_freqs`` holding
its postings. Loading is therefore a handful of array reads rather than a walk
over a nested Python structure, which keeps the retriever's cold start short.

Alongside it, a chunk-major document cache keeps the tokenized postings grouped
by source file together with the file's modification stamp. Re-indexing can
then reuse untouched files and only re-chunk the ones that actually changed.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .chunking import CODE_SUFFIXES, TEXT_SUFFIXES, chunk_file
from .tokenization import iter_tokens

DEFAULT_SUFFIXES: Tuple[str, ...] = tuple(sorted(CODE_SUFFIXES | TEXT_SUFFIXES))
DEFAULT_MAX_CHUNK_SIZE = 2000
CONTEXT_WEIGHT = 3.0
ARRAYS_FILE = "index.npz"
VOCAB_FILE = "vocabulary.json"
META_FILE = "metadata.json"
CACHE_ARRAYS_FILE = "documents.npz"
CACHE_META_FILE = "documents.json"

ScalarT = TypeVar("ScalarT", bound=np.generic)


class IndexError_(RuntimeError):
    """Raised when an index cannot be built or loaded."""


@dataclass
class FileEntry:
    """The tokenized contribution of a single corpus file.

    Attributes:
        path: Corpus-relative path of the file.
        mtime_ns: Modification time used to detect changes.
        size: File size in bytes, used to detect changes.
        chunk_first: Start offset of every chunk of the file.
        chunk_last: End offset of every chunk of the file.
        chunk_sizes: Number of postings contributed by every chunk.
        term_ids: Term identifiers, grouped by chunk.
        weights: Weighted term frequencies, aligned with ``term_ids``.
    """

    path: str
    mtime_ns: int
    size: int
    chunk_first: NDArray[np.int32]
    chunk_last: NDArray[np.int32]
    chunk_sizes: NDArray[np.int32]
    term_ids: NDArray[np.int32]
    weights: NDArray[np.float32]

    @property
    def chunk_count(self) -> int:
        """Number of chunks the file contributes."""
        return int(self.chunk_first.shape[0])


@dataclass
class Index:
    """An inverted index over the chunked corpus.

    Attributes:
        vocabulary: Term to term identifier.
        indptr: Postings slice boundaries, one entry per term plus one.
        doc_ids: Chunk identifiers, grouped by term.
        term_freqs: Weighted term frequencies, aligned with ``doc_ids``.
        chunk_lens: Weighted length of every chunk.
        file_paths: Distinct corpus-relative file paths.
        chunk_files: Index into ``file_paths`` for every chunk.
        chunk_first: Start offset of every chunk.
        chunk_last: End offset of every chunk.
        max_chunk_size: Character cap the index was built with.
    """

    vocabulary: Dict[str, int]
    indptr: NDArray[np.int64]
    doc_ids: NDArray[np.int32]
    term_freqs: NDArray[np.float32]
    chunk_lens: NDArray[np.float32]
    file_paths: List[str]
    chunk_files: NDArray[np.int32]
    chunk_first: NDArray[np.int32]
    chunk_last: NDArray[np.int32]
    max_chunk_size: int

    @property
    def chunk_count(self) -> int:
        """Number of chunks held by the index."""
        return int(self.chunk_lens.shape[0])

    def postings(
        self, term: str
    ) -> Optional[Tuple[NDArray[np.int32], NDArray[np.float32]]]:
        """Return the postings of ``term``.

        Args:
            term: The term to look up.

        Returns:
            The chunk identifiers and weighted frequencies, or None when the
            term is absent from the vocabulary.
        """
        term_id = self.vocabulary.get(term)
        if term_id is None:
            return None
        start, end = int(self.indptr[term_id]), int(self.indptr[term_id + 1])
        return self.doc_ids[start:end], self.term_freqs[start:end]

    def save(self, directory: str) -> None:
        """Persist the index under ``directory``.

        Args:
            directory: Destination directory, created when missing.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.savez(
            path / ARRAYS_FILE,
            indptr=self.indptr,
            doc_ids=self.doc_ids,
            term_freqs=self.term_freqs,
            chunk_lens=self.chunk_lens,
            chunk_files=self.chunk_files,
            chunk_first=self.chunk_first,
            chunk_last=self.chunk_last,
        )
        with open(path / VOCAB_FILE, "w", encoding="utf-8") as handle:
            json.dump(self.vocabulary, handle)
        with open(path / META_FILE, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "file_paths": self.file_paths,
                    "max_chunk_size": self.max_chunk_size,
                    "chunk_count": self.chunk_count,
                },
                handle,
            )

    @classmethod
    def load(cls, directory: str, mmap: bool = True) -> "Index":
        """Load an index previously written by :meth:`save`.

        Args:
            directory: Directory holding the index files.
            mmap: Whether to memory-map the arrays instead of reading them,
                which keeps the cold start short for a single query.

        Returns:
            The loaded index.

        Raises:
            IndexError_: When the directory holds no readable index.
        """
        path = Path(directory)
        try:
            arrays = _load_arrays(path / ARRAYS_FILE, mmap)
            with open(path / VOCAB_FILE, encoding="utf-8") as handle:
                vocabulary = json.load(handle)
            with open(path / META_FILE, encoding="utf-8") as handle:
                meta = json.load(handle)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise IndexError_(
                f"no usable index in '{directory}' ({error}); "
                "run the index command first"
            ) from error
        return cls(
            vocabulary=vocabulary,
            indptr=arrays["indptr"],
            doc_ids=arrays["doc_ids"],
            term_freqs=arrays["term_freqs"],
            chunk_lens=arrays["chunk_lens"],
            file_paths=list(meta["file_paths"]),
            chunk_files=arrays["chunk_files"],
            chunk_first=arrays["chunk_first"],
            chunk_last=arrays["chunk_last"],
            max_chunk_size=int(meta["max_chunk_size"]),
        )


def _load_arrays(path: Path, mmap: bool) -> Dict[str, NDArray[Any]]:
    """Read a ``.npz`` archive, memory-mapping it when asked.

    Args:
        path: Path of the archive.
        mmap: Whether to memory-map the arrays.

    Returns:
        The arrays it holds, keyed by name.
    """
    if mmap:
        # An uncompressed npz memory-maps entry by entry; scoring only touches
        # the postings slices it needs rather than the whole 12 MB archive.
        with np.load(path, mmap_mode="r") as archive:
            return {name: archive[name] for name in archive.files}
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


def iter_corpus_files(
    raw_directory: str, suffixes: Sequence[str] = DEFAULT_SUFFIXES
) -> Iterator[Tuple[str, str]]:
    """Walk ``raw_directory`` and yield the files worth indexing.

    Args:
        raw_directory: Root of the ingested corpus.
        suffixes: Lower-cased extensions to keep, dot included.

    Yields:
        Pairs of corpus-relative path and lower-cased extension, sorted so
        that chunk identifiers stay stable across rebuilds.
    """
    wanted = frozenset(suffixes)
    for directory, subdirectories, names in os.walk(raw_directory):
        subdirectories.sort()
        for name in sorted(names):
            suffix = os.path.splitext(name)[1].lower()
            if suffix in wanted:
                yield os.path.join(directory, name), suffix


def read_text(path: str) -> Optional[str]:
    """Read a corpus file as UTF-8 text.

    Args:
        path: Path of the file to read.

    Returns:
        The decoded text, or None when the file cannot be read.
    """
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            return handle.read()
    except OSError:
        return None


def _tokenize_file(
    path: str,
    suffix: str,
    text: str,
    max_chunk_size: int,
    vocabulary: Dict[str, int],
    stamp: Tuple[int, int],
) -> Optional[FileEntry]:
    """Chunk and tokenize one file, extending the vocabulary as needed.

    Args:
        path: Corpus-relative path of the file.
        suffix: Lower-cased file extension, dot included.
        text: The full file text.
        max_chunk_size: Maximum chunk length in characters.
        vocabulary: The append-only term table, updated in place.
        stamp: The file's ``(mtime_ns, size)`` pair.

    Returns:
        The file's contribution, or None when it chunks to nothing.
    """
    try:
        chunks = chunk_file(path, suffix, text, max_chunk_size)
    except RecursionError:
        return None
    if not chunks:
        return None
    term_ids: List[int] = []
    weights: List[float] = []
    sizes: List[int] = []
    for chunk in chunks:
        scored: Dict[int, float] = {}
        for term in iter_tokens(chunk.body):
            term_id = vocabulary.setdefault(term, len(vocabulary))
            scored[term_id] = scored.get(term_id, 0.0) + 1.0
        for term in iter_tokens(chunk.context):
            term_id = vocabulary.setdefault(term, len(vocabulary))
            scored[term_id] = scored.get(term_id, 0.0) + CONTEXT_WEIGHT
        term_ids.extend(scored)
        weights.extend(scored.values())
        sizes.append(len(scored))
    return FileEntry(
        path=path,
        mtime_ns=stamp[0],
        size=stamp[1],
        chunk_first=np.asarray(
            [chunk.first_character_index for chunk in chunks], dtype=np.int32
        ),
        chunk_last=np.asarray(
            [chunk.last_character_index for chunk in chunks], dtype=np.int32
        ),
        chunk_sizes=np.asarray(sizes, dtype=np.int32),
        term_ids=np.asarray(term_ids, dtype=np.int32),
        weights=np.asarray(weights, dtype=np.float32),
    )


@dataclass
class DocumentCache:
    """Per-file tokenized postings, enabling incremental re-indexing.

    Attributes:
        vocabulary: The append-only term table shared by every entry.
        entries: One entry per indexed file, in corpus order.
        max_chunk_size: Character cap the entries were built with.
    """

    vocabulary: Dict[str, int]
    entries: List[FileEntry]
    max_chunk_size: int

    def save(self, directory: str) -> None:
        """Persist the cache under ``directory``.

        Args:
            directory: Destination directory, created when missing.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        empty_int = np.zeros(0, dtype=np.int32)
        empty_float = np.zeros(0, dtype=np.float32)
        np.savez(
            path / CACHE_ARRAYS_FILE,
            file_chunk_counts=np.asarray(
                [entry.chunk_count for entry in self.entries], dtype=np.int64
            ),
            chunk_first=_concat(
                [entry.chunk_first for entry in self.entries], empty_int
            ),
            chunk_last=_concat([entry.chunk_last for entry in self.entries], empty_int),
            chunk_sizes=_concat(
                [entry.chunk_sizes for entry in self.entries], empty_int
            ),
            term_ids=_concat([entry.term_ids for entry in self.entries], empty_int),
            weights=_concat([entry.weights for entry in self.entries], empty_float),
        )
        with open(path / CACHE_META_FILE, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "max_chunk_size": self.max_chunk_size,
                    "files": [
                        [entry.path, entry.mtime_ns, entry.size]
                        for entry in self.entries
                    ],
                },
                handle,
            )

    @classmethod
    def load(cls, directory: str) -> Optional["DocumentCache"]:
        """Load a cache previously written by :meth:`save`.

        Args:
            directory: Directory holding the cache files.

        Returns:
            The cache, or None when none is readable. A missing or damaged
            cache is not an error: it only means a full rebuild is needed.
        """
        path = Path(directory)
        try:
            arrays = _load_arrays(path / CACHE_ARRAYS_FILE, mmap=False)
            with open(path / CACHE_META_FILE, encoding="utf-8") as handle:
                meta = json.load(handle)
            with open(path / VOCAB_FILE, encoding="utf-8") as handle:
                vocabulary = json.load(handle)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None
        entries: List[FileEntry] = []
        chunk_start = 0
        posting_start = 0
        chunk_sizes = arrays["chunk_sizes"]
        try:
            for (file_path, mtime_ns, size), count in zip(
                meta["files"], arrays["file_chunk_counts"]
            ):
                chunk_end = chunk_start + int(count)
                span = chunk_sizes[chunk_start:chunk_end]
                posting_end = posting_start + int(span.sum())
                entries.append(
                    FileEntry(
                        path=file_path,
                        mtime_ns=int(mtime_ns),
                        size=int(size),
                        chunk_first=arrays["chunk_first"][chunk_start:chunk_end],
                        chunk_last=arrays["chunk_last"][chunk_start:chunk_end],
                        chunk_sizes=span,
                        term_ids=arrays["term_ids"][posting_start:posting_end],
                        weights=arrays["weights"][posting_start:posting_end],
                    )
                )
                chunk_start, posting_start = chunk_end, posting_end
        except (KeyError, TypeError, ValueError):
            return None
        return cls(vocabulary, entries, int(meta["max_chunk_size"]))

    def to_index(self) -> Index:
        """Assemble the term-major index from the cached per-file postings.

        Returns:
            The inverted index.

        Raises:
            IndexError_: When the cache holds no chunk.
        """
        chunk_counts = [entry.chunk_count for entry in self.entries]
        total_chunks = sum(chunk_counts)
        if not total_chunks:
            raise IndexError_("the document cache holds no chunk to index")
        term_ids = _concat(
            [entry.term_ids for entry in self.entries], np.zeros(0, dtype=np.int32)
        ).astype(np.int64)
        weights = _concat(
            [entry.weights for entry in self.entries], np.zeros(0, dtype=np.float32)
        )
        chunk_sizes = _concat(
            [entry.chunk_sizes for entry in self.entries], np.zeros(0, dtype=np.int32)
        )
        chunk_ids = np.repeat(
            np.arange(total_chunks, dtype=np.int32), chunk_sizes.astype(np.int64)
        )
        chunk_files = np.repeat(
            np.arange(len(self.entries), dtype=np.int32),
            np.asarray(chunk_counts, dtype=np.int64),
        )
        chunk_lens = np.bincount(
            chunk_ids, weights=weights, minlength=total_chunks
        ).astype(np.float32)
        order = np.argsort(term_ids, kind="stable")
        counts = np.bincount(term_ids, minlength=len(self.vocabulary))
        indptr = np.zeros(len(self.vocabulary) + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        return Index(
            vocabulary=self.vocabulary,
            indptr=indptr,
            doc_ids=chunk_ids[order],
            term_freqs=weights[order],
            chunk_lens=chunk_lens,
            file_paths=[entry.path for entry in self.entries],
            chunk_files=chunk_files,
            chunk_first=_concat(
                [entry.chunk_first for entry in self.entries],
                np.zeros(0, dtype=np.int32),
            ),
            chunk_last=_concat(
                [entry.chunk_last for entry in self.entries],
                np.zeros(0, dtype=np.int32),
            ),
            max_chunk_size=self.max_chunk_size,
        )


def _concat(
    arrays: Sequence[NDArray[ScalarT]], empty: NDArray[ScalarT]
) -> NDArray[ScalarT]:
    """Concatenate arrays, tolerating an empty sequence.

    Args:
        arrays: The arrays to concatenate.
        empty: The array returned when there is nothing to concatenate.

    Returns:
        The concatenation.
    """
    return np.concatenate(arrays) if arrays else empty


def build_document_cache(
    raw_directory: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    suffixes: Sequence[str] = DEFAULT_SUFFIXES,
    previous: Optional[DocumentCache] = None,
    show_progress: bool = True,
) -> Tuple[DocumentCache, int]:
    """Chunk and tokenize the corpus, reusing unchanged files when possible.

    A file is reused when its path, modification time and size all match the
    previous cache and the chunk size has not changed. Deleted files simply
    drop out, since the walk drives the result.

    Args:
        raw_directory: Root of the ingested corpus.
        max_chunk_size: Maximum chunk length in characters.
        suffixes: Lower-cased extensions to keep, dot included.
        previous: A cache to reuse entries from, when compatible.
        show_progress: Whether to display a progress bar.

    Returns:
        The new cache and the number of entries reused from ``previous``.

    Raises:
        IndexError_: When the corpus holds no indexable file.
    """
    files = list(iter_corpus_files(raw_directory, suffixes))
    if not files:
        raise IndexError_(f"no indexable file found under '{raw_directory}'")

    reusable: Dict[str, FileEntry] = {}
    vocabulary: Dict[str, int] = {}
    if previous is not None and previous.max_chunk_size == max_chunk_size:
        reusable = {entry.path: entry for entry in previous.entries}
        vocabulary = previous.vocabulary

    entries: List[FileEntry] = []
    reused = 0
    for path, suffix in tqdm(
        files, desc="Chunking", unit="file", disable=not show_progress
    ):
        try:
            status = os.stat(path)
            stamp = (status.st_mtime_ns, status.st_size)
        except OSError:
            continue
        cached = reusable.get(path)
        if cached is not None and (cached.mtime_ns, cached.size) == stamp:
            entries.append(cached)
            reused += 1
            continue
        text = read_text(path)
        if not text:
            continue
        entry = _tokenize_file(path, suffix, text, max_chunk_size, vocabulary, stamp)
        if entry is not None:
            entries.append(entry)
    if not entries:
        raise IndexError_(f"every file under '{raw_directory}' chunked to nothing")
    return DocumentCache(vocabulary, entries, max_chunk_size), reused


def build_index(
    raw_directory: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    suffixes: Sequence[str] = DEFAULT_SUFFIXES,
    show_progress: bool = True,
) -> Index:
    """Ingest a corpus from scratch and build its inverted index.

    Args:
        raw_directory: Root of the ingested corpus.
        max_chunk_size: Maximum chunk length in characters.
        suffixes: Lower-cased extensions to keep, dot included.
        show_progress: Whether to display progress bars.

    Returns:
        The freshly built index.

    Raises:
        IndexError_: When the corpus yields nothing to index.
    """
    cache, _ = build_document_cache(
        raw_directory, max_chunk_size, suffixes, None, show_progress
    )
    return cache.to_index()
