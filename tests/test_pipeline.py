"""Tests for tokenization, indexing, retrieval and evaluation."""

import json
from pathlib import Path

import pytest

from src.evaluation import intersection_over_union, recall_at_k
from src.indexing import (
    DocumentCache,
    IndexError_,
    Index,
    build_document_cache,
    build_index,
)
from src.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.pipeline import PipelineError, load_rag_dataset, search_dataset
from src.retrieval import Retriever
from src.tokenization import tokenize


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """Build a tiny two-file corpus on disk."""
    root = tmp_path / "raw"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "serving.md").write_text(
        "# Serving\n\nRun the OpenAI compatible server with vllm serve.\n",
        encoding="utf-8",
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
    return build_index(str(corpus), max_chunk_size=2000, show_progress=False)


def test_tokenizer_splits_compound_identifiers() -> None:
    """A snake_case identifier must match both as a whole and by sub-word."""
    terms = tokenize("load_lora_adapter")
    assert "load_lora_adapter" in terms
    assert {"load", "lora", "adapter"} <= set(terms)


def test_tokenizer_splits_camel_case_and_folds_plurals() -> None:
    """Camel case must break apart and simple plurals must fold."""
    assert {"async", "llm", "engine"} <= set(tokenize("AsyncLLMEngine"))
    assert tokenize("endpoints") == tokenize("endpoint")


def test_tokenizer_ignores_stopwords_and_punctuation() -> None:
    """Noise words must not reach the index."""
    assert tokenize("the and of ---") == []


def test_index_round_trips_through_disk(index: Index, tmp_path: Path) -> None:
    """A saved index must reload identically."""
    index.save(str(tmp_path / "processed"))
    reloaded = Index.load(str(tmp_path / "processed"))
    assert reloaded.chunk_count == index.chunk_count
    assert reloaded.file_paths == index.file_paths
    assert reloaded.vocabulary == index.vocabulary


def test_loading_a_missing_index_raises_cleanly(tmp_path: Path) -> None:
    """A missing index must surface as a domain error, not an OSError."""
    with pytest.raises(IndexError_):
        Index.load(str(tmp_path / "absent"))


def test_indexing_an_empty_corpus_raises_cleanly(tmp_path: Path) -> None:
    """A corpus without indexable files must surface as a domain error."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(IndexError_):
        build_index(str(tmp_path / "empty"), show_progress=False)


def _as_tuples(index: Index) -> object:
    """Reduce an index to a comparable plain-Python form."""
    return (
        index.vocabulary,
        index.file_paths,
        index.indptr.tolist(),
        index.doc_ids.tolist(),
        index.term_freqs.tolist(),
        index.chunk_lens.tolist(),
        index.chunk_files.tolist(),
        index.chunk_first.tolist(),
        index.chunk_last.tolist(),
    )


def test_incremental_indexing_reuses_untouched_files(corpus: Path) -> None:
    """An unchanged corpus must be fully served from the document cache."""
    first, reused = build_document_cache(str(corpus), show_progress=False)
    assert reused == 0
    second, reused = build_document_cache(
        str(corpus), previous=first, show_progress=False
    )
    assert reused == len(first.entries)
    assert _as_tuples(second.to_index()) == _as_tuples(first.to_index())


def test_incremental_index_matches_a_full_rebuild(corpus: Path) -> None:
    """Editing one file must give the same index as rebuilding from scratch."""
    cached, _ = build_document_cache(str(corpus), show_progress=False)
    (corpus / "docs" / "serving.md").write_text(
        "# Serving\n\nStart the tensor parallel worker pool.\n", encoding="utf-8"
    )
    incremental, reused = build_document_cache(
        str(corpus), previous=cached, show_progress=False
    )
    assert reused == len(cached.entries) - 1
    rebuilt = build_index(str(corpus), show_progress=False)
    incremental_index = incremental.to_index()
    assert incremental_index.file_paths == rebuilt.file_paths
    assert incremental_index.chunk_first.tolist() == rebuilt.chunk_first.tolist()
    assert incremental_index.chunk_lens.tolist() == rebuilt.chunk_lens.tolist()
    hits = Retriever(incremental_index).search("tensor parallel worker", k=5)
    assert hits and hits[0].file_path.endswith("serving.md")


def test_document_cache_round_trips_through_disk(
    corpus: Path, tmp_path: Path
) -> None:
    """A reloaded cache must rebuild exactly the same index."""
    cache, _ = build_document_cache(str(corpus), show_progress=False)
    directory = str(tmp_path / "processed")
    cache.to_index().save(directory)
    cache.save(directory)
    reloaded = DocumentCache.load(directory)
    assert reloaded is not None
    assert _as_tuples(reloaded.to_index()) == _as_tuples(cache.to_index())


def test_missing_document_cache_is_not_an_error(tmp_path: Path) -> None:
    """A first run has no cache to load and must simply get None."""
    assert DocumentCache.load(str(tmp_path / "absent")) is None


def test_repeated_queries_are_served_from_the_cache(index: Index) -> None:
    """A second identical query must not rescore the corpus."""
    retriever = Retriever(index)
    first = retriever.search("lora adapter", k=5)
    retriever.score = None  # type: ignore[assignment]
    assert retriever.search("lora adapter", k=5) == first


def test_retrieval_ranks_the_relevant_file_first(index: Index) -> None:
    """A question phrased in prose must reach the matching source."""
    sources = Retriever(index).search("How do I load a LoRA adapter?", k=5)
    assert sources
    assert sources[0].file_path.endswith("engine.py")


def test_retrieval_handles_degenerate_queries(index: Index) -> None:
    """Empty, unknown and zero-k queries must return nothing, not raise."""
    retriever = Retriever(index)
    assert retriever.search("", k=5) == []
    assert retriever.search("zzqqxx flurble", k=5) == []
    assert retriever.search("lora adapter", k=0) == []


def test_retrieved_spans_respect_the_chunk_size_cap(corpus: Path) -> None:
    """No retrieved source may exceed the configured chunk size."""
    small = build_index(str(corpus), max_chunk_size=64, show_progress=False)
    for source in Retriever(small).search("lora adapter server", k=10):
        span = source.last_character_index - source.first_character_index
        assert span <= 64


def test_intersection_over_union_needs_the_same_file() -> None:
    """Spans of different files never overlap, however close their offsets."""
    left = MinimalSource(
        file_path="a.md", first_character_index=0, last_character_index=100
    )
    right = MinimalSource(
        file_path="b.md", first_character_index=0, last_character_index=100
    )
    assert intersection_over_union(left, right) == 0.0
    assert intersection_over_union(left, left) == 1.0


def test_recall_counts_a_partial_overlap_as_found() -> None:
    """A chunk covering part of the reference span must count as retrieved."""
    reference = MinimalSource(
        file_path="a.md", first_character_index=1000, last_character_index=2000
    )
    retrieved = MinimalSource(
        file_path="a.md", first_character_index=1500, last_character_index=2500
    )
    dataset = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1", question="?", sources=[reference], answer="a"
            )
        ]
    )
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q1", question="?", retrieved_sources=[retrieved]
            )
        ],
        k=10,
    )
    assert recall_at_k(results, dataset, cutoffs=(1,))[1] == 1.0


def test_recall_is_zero_without_any_retrieval() -> None:
    """A question with no retrieved source must score zero, not crash."""
    dataset = RagDataset(
        rag_questions=[
            AnsweredQuestion(
                question_id="q1",
                question="?",
                sources=[
                    MinimalSource(
                        file_path="a.md",
                        first_character_index=0,
                        last_character_index=10,
                    )
                ],
                answer="a",
            )
        ]
    )
    results = StudentSearchResults(search_results=[], k=10)
    assert recall_at_k(results, dataset, cutoffs=(1, 5))[5] == 0.0


def test_search_dataset_answers_every_question(index: Index, tmp_path: Path) -> None:
    """Batch search must emit one entry per question, ids preserved."""
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {"question_id": "q1", "question": "How to serve a model?"},
                    {"question_id": "q2", "question": "How to load a LoRA adapter?"},
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = load_rag_dataset(str(path))
    results = search_dataset(Retriever(index), dataset, k=3, show_progress=False)
    assert [result.question_id for result in results.search_results] == ["q1", "q2"]
    assert results.k == 3


def test_malformed_dataset_raises_a_pipeline_error(tmp_path: Path) -> None:
    """Broken JSON must never reach the caller as a JSONDecodeError."""
    path = tmp_path / "broken.json"
    path.write_text('{"rag_questions": [', encoding="utf-8")
    with pytest.raises(PipelineError):
        load_rag_dataset(str(path))


def test_missing_dataset_raises_a_pipeline_error(tmp_path: Path) -> None:
    """A missing file must never reach the caller as a FileNotFoundError."""
    with pytest.raises(PipelineError):
        load_rag_dataset(str(tmp_path / "absent.json"))
