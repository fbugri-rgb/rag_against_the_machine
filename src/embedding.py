"""Dense chunk embeddings backing the optional hybrid retriever.

The lexical index stays the source of truth: this module only adds a vector
per chunk so that a semantic score can be fused into the BM25 ranking. Chunk
text is reconstructed from the corpus by slicing the offsets the index already
stores, which keeps the embeddings in step with the chunking strategy without
duplicating the corpus on disk.
"""

from pathlib import Path
from typing import Any, Iterator, List, Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from .indexing import Index

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 64
VECTORS_FILE = "embeddings.npy"


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be built or loaded."""


def iter_chunk_texts(index: Index) -> Iterator[str]:
    """Rebuild the text of every chunk, in chunk identifier order.

    The corpus-relative path is prefixed to the body: it is short, highly
    discriminative (``docs/source/features/lora.md``) and free, since the
    index already holds it.

    Args:
        index: The lexical index describing the chunks.

    Yields:
        The text to embed for each chunk.

    Raises:
        EmbeddingError: When a corpus file can no longer be read.
    """
    last_path = ""
    text = ""
    for chunk_id in range(index.chunk_count):
        path = index.file_paths[int(index.chunk_files[chunk_id])]
        if path != last_path:
            try:
                with open(path, encoding="utf-8", errors="replace", newline="") as file:
                    text = file.read()
            except OSError as error:
                raise EmbeddingError(f"cannot re-read '{path}': {error}") from error
            last_path = path
        start = int(index.chunk_first[chunk_id])
        end = int(index.chunk_last[chunk_id])
        yield f"{path}\n{text[start:end]}"


def _load_model(model_name: str) -> Any:
    """Instantiate the sentence transformer behind a friendly error.

    Args:
        model_name: Identifier of the embedding model.

    Returns:
        The loaded model.

    Raises:
        EmbeddingError: When the model cannot be imported or downloaded.
    """
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as error:  # noqa: BLE001 - surfaced as a domain error
        raise EmbeddingError(
            f"cannot load the embedding model '{model_name}': {error}"
        ) from error


class EmbeddingIndex:
    """Unit-norm vectors for every chunk of a lexical index.

    Attributes:
        vectors: One row per chunk, L2-normalised.
        model_name: Identifier of the model that produced them.
    """

    def __init__(self, vectors: NDArray[np.float32], model_name: str) -> None:
        """Store the vectors and the model they came from.

        Args:
            vectors: One row per chunk, L2-normalised.
            model_name: Identifier of the embedding model.
        """
        self.vectors = vectors
        self.model_name = model_name
        self._model: Optional[Any] = None

    @property
    def chunk_count(self) -> int:
        """Number of chunks the vectors cover."""
        return int(self.vectors.shape[0])

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Embed a query into the chunk vector space.

        Args:
            query: The natural-language query.

        Returns:
            The unit-norm query vector.

        Raises:
            EmbeddingError: When the model cannot be loaded.
        """
        if self._model is None:
            self._model = _load_model(self.model_name)
        vector = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector[0], dtype=np.float32)

    def save(self, directory: str) -> None:
        """Persist the vectors under ``directory``.

        Args:
            directory: Destination directory, created when missing.
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / VECTORS_FILE, self.vectors)

    @classmethod
    def load(cls, directory: str, model_name: str) -> Optional["EmbeddingIndex"]:
        """Load vectors previously written by :meth:`save`.

        Args:
            directory: Directory holding the vectors.
            model_name: Identifier of the model to encode queries with.

        Returns:
            The embedding index, or None when none is readable. Absence is
            not an error: retrieval simply stays lexical.
        """
        try:
            vectors = np.load(Path(directory) / VECTORS_FILE)
        except (OSError, ValueError):
            return None
        return cls(vectors.astype(np.float32), model_name)


def build_embedding_index(
    index: Index,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> EmbeddingIndex:
    """Embed every chunk of ``index``.

    Args:
        index: The lexical index describing the chunks.
        model_name: Identifier of the embedding model.
        batch_size: Number of chunks encoded per forward pass.
        show_progress: Whether to display a progress bar.

    Returns:
        The freshly built embedding index.

    Raises:
        EmbeddingError: When the model or the corpus cannot be read.
    """
    model = _load_model(model_name)
    texts = list(iter_chunk_texts(index))
    batches: List[NDArray[np.float32]] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding",
        unit="batch",
        disable=not show_progress,
    ):
        batch = texts[start : start + batch_size]  # noqa: E203
        batches.append(
            model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)
        )
    vectors = np.concatenate(batches) if batches else np.zeros((0, 0), np.float32)
    return EmbeddingIndex(vectors, model_name)
