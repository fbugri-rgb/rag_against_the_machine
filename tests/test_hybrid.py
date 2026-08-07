"""Tests for chunk embeddings and hybrid reranking.

The embedding model itself is never loaded here: the vectors are synthesised
so the fusion arithmetic can be checked deterministically and offline.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.embedding import EmbeddingIndex, iter_chunk_texts
from src.indexing import Index, build_index
from src.retrieval import HybridRetriever


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Build a tiny two-file corpus on disk."""
    root = tmp_path / "raw"
    root.mkdir()
    (root / "serving.md").write_text(
        "# Serving\n\nRun the OpenAI compatible server.\n", encoding="utf-8"
    )
    (root / "engine.py").write_text(
        'def load_lora_adapter(name: str) -> None:\n'
        '    """Load a LoRA adapter by name."""\n'
        "    return None\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture()
def index(corpus: Path) -> Index:
    """Build an index over the tiny corpus."""
    return build_index(str(corpus), show_progress=False)


def _fake_embeddings(vectors: NDArray[Any]) -> EmbeddingIndex:
    """Wrap raw vectors as an embedding index, bypassing the real model."""
    return EmbeddingIndex(vectors.astype(np.float32), "test-model")


def test_chunk_texts_carry_the_file_path(index: Index) -> None:
    """The reconstructed text must prefix the path to a verbatim slice."""
    texts = list(iter_chunk_texts(index))
    assert len(texts) == index.chunk_count
    for chunk_id, text in enumerate(texts):
        path = index.file_paths[int(index.chunk_files[chunk_id])]
        body = Path(path).read_text(encoding="utf-8")
        start = int(index.chunk_first[chunk_id])
        end = int(index.chunk_last[chunk_id])
        assert text == f"{path}\n{body[start:end]}"


def test_embeddings_round_trip_through_disk(index: Index, tmp_path: Path) -> None:
    """Saved vectors must reload unchanged."""
    vectors = np.random.default_rng(0).normal(size=(index.chunk_count, 8))
    _fake_embeddings(vectors).save(str(tmp_path))
    reloaded = EmbeddingIndex.load(str(tmp_path), "test-model")
    assert reloaded is not None
    assert reloaded.chunk_count == index.chunk_count
    assert np.allclose(reloaded.vectors, vectors.astype(np.float32))


def test_missing_embeddings_are_not_an_error(tmp_path: Path) -> None:
    """A corpus indexed without embeddings must simply yield None."""
    assert EmbeddingIndex.load(str(tmp_path / "absent"), "test-model") is None


def test_hybrid_rejects_vectors_that_do_not_cover_the_index(index: Index) -> None:
    """Stale vectors must be refused rather than silently misaligned."""
    short = _fake_embeddings(np.zeros((index.chunk_count - 1, 4)))
    with pytest.raises(ValueError):
        HybridRetriever(index, short)


def test_hybrid_reranking_can_promote_a_semantic_match(index: Index) -> None:
    """With no lexical weight the dense score alone must decide the order."""
    vectors = np.zeros((index.chunk_count, 2), dtype=np.float32)
    vectors[:, 0] = 1.0
    target = index.chunk_count - 1
    vectors[target] = (0.0, 1.0)
    embeddings = _fake_embeddings(vectors)
    embeddings.encode_query = lambda query: np.asarray(  # type: ignore[method-assign]
        [0.0, 1.0], dtype=np.float32
    )
    retriever = HybridRetriever(index, embeddings, alpha=0.0)
    sources = retriever.search("lora adapter server", k=1)
    assert sources == [retriever.source_of(target)]


def test_hybrid_falls_back_to_nothing_on_an_unknown_query(index: Index) -> None:
    """A query with no lexical match must not be rescued by the vectors."""
    embeddings = _fake_embeddings(np.ones((index.chunk_count, 2)))
    assert HybridRetriever(index, embeddings).search("zzqqxx flurble", k=5) == []
