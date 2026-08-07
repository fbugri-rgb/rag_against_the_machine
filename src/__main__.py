"""Command-line entry point of the RAG pipeline."""

import os
import sys
import time
from typing import List, NoReturn, Optional

import fire
from tqdm import tqdm

from .embedding import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingError,
    build_embedding_index,
)
from .evaluation import format_report, recall_at_k
from .generation import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_SOURCES,
    DEFAULT_MODEL,
    AnswerGenerator,
    GenerationError,
)
from .indexing import (
    DEFAULT_MAX_CHUNK_SIZE,
    DocumentCache,
    IndexError_,
    build_document_cache,
)
from .models import MinimalAnswer, MinimalSource, StudentSearchResultsAndAnswer
from .pipeline import (
    PipelineError,
    load_rag_dataset,
    load_student_search_results,
    search_dataset,
    write_model,
)
from .retrieval import HYBRID_ALPHA, Retriever, load_retriever

DEFAULT_RAW_DIRECTORY = "data/raw"
DEFAULT_INDEX_DIRECTORY = "data/processed"
DEFAULT_SEARCH_DIRECTORY = "data/output/search_results"
DEFAULT_ANSWER_DIRECTORY = "data/output/search_results_and_answer"
DEFAULT_K = 10


def _abort(message: str) -> NoReturn:
    """Report a user-facing error and stop with a non-zero status.

    Args:
        message: The error to report.

    Raises:
        SystemExit: Always.
    """
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(1)


def _retriever(
    index_directory: str, hybrid: bool = False, alpha: float = HYBRID_ALPHA
) -> Retriever:
    """Load the retriever, turning a missing index into a clean error.

    Args:
        index_directory: Directory holding the persisted index.
        hybrid: Whether to rerank the lexical shortlist with embeddings.
        alpha: Weight of the lexical score in the fusion.

    Returns:
        The loaded retriever.
    """
    try:
        retriever, _ = load_retriever(index_directory, hybrid, alpha)
    except IndexError_ as error:
        _abort(str(error))
    return retriever


def _print_sources(sources: List[MinimalSource]) -> None:
    """Print retrieved source locations, one per line.

    Args:
        sources: The source locations to print.
    """
    if not sources:
        print("No matching source found.")
        return
    for source in sources:
        span = f"[{source.first_character_index}:{source.last_character_index}]"
        print(f"{source.file_path} {span}")


class RagCli:
    """The commands exposed by ``python -m src``."""

    def index(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        raw_directory: str = DEFAULT_RAW_DIRECTORY,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        rebuild: bool = False,
        embeddings: bool = False,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Ingest the corpus and persist its index.

        Files whose modification time and size are unchanged since the last
        run are reused from the document cache instead of being re-chunked.

        Args:
            max_chunk_size: Maximum chunk length in characters.
            raw_directory: Root of the corpus to ingest.
            index_directory: Directory the index is written to.
            rebuild: Ignore the document cache and re-read every file.
            embeddings: Also embed every chunk, enabling hybrid retrieval.
                This is off by default because it costs several minutes.
            embedding_model: Identifier of the embedding model.
        """
        if max_chunk_size <= 0:
            _abort("--max_chunk_size must be a positive number of characters")
        started = time.perf_counter()
        previous = None if rebuild else DocumentCache.load(index_directory)
        try:
            cache, reused = build_document_cache(
                raw_directory, max_chunk_size, previous=previous
            )
            index = cache.to_index()
            index.save(index_directory)
            cache.save(index_directory)
        except IndexError_ as error:
            _abort(str(error))
        except OSError as error:
            _abort(f"cannot write the index to '{index_directory}': {error}")
        elapsed = time.perf_counter() - started
        print(
            f"Ingestion complete! Indexed {index.chunk_count} chunks "
            f"from {len(index.file_paths)} files under {index_directory}/ "
            f"in {elapsed:.1f}s ({reused} files reused from cache)"
        )
        if embeddings:
            try:
                build_embedding_index(index, embedding_model).save(index_directory)
            except EmbeddingError as error:
                _abort(str(error))
            except OSError as error:
                _abort(f"cannot write the embeddings to '{index_directory}': {error}")
            print(f"Embedded {index.chunk_count} chunks with {embedding_model}")

    def search(
        self,
        query: str = "",
        k: int = DEFAULT_K,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        hybrid: bool = False,
        alpha: float = HYBRID_ALPHA,
    ) -> None:
        """Return the top-k sources for a single query.

        Args:
            query: The natural-language query.
            k: Maximum number of sources to return.
            index_directory: Directory holding the persisted index.
            hybrid: Rerank the lexical shortlist with chunk embeddings.
            alpha: Weight of the lexical score in the fusion.
        """
        query = str(query).strip()
        if not query:
            _abort("a non-empty query is required")
        retriever = _retriever(index_directory, hybrid, alpha)
        _print_sources(retriever.search(query, max(k, 0)))

    def search_dataset(
        self,
        dataset_path: str = "",
        k: int = DEFAULT_K,
        save_directory: str = DEFAULT_SEARCH_DIRECTORY,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        hybrid: bool = False,
        alpha: float = HYBRID_ALPHA,
    ) -> None:
        """Run retrieval over a whole dataset and save the results.

        Args:
            dataset_path: Path of the dataset JSON file.
            k: Maximum number of sources per question.
            save_directory: Directory the results are written to.
            index_directory: Directory holding the persisted index.
            hybrid: Rerank the lexical shortlist with chunk embeddings.
            alpha: Weight of the lexical score in the fusion.
        """
        if not dataset_path:
            _abort("--dataset_path is required")
        retriever = _retriever(index_directory, hybrid, alpha)
        try:
            dataset = load_rag_dataset(dataset_path)
            results = search_dataset(retriever, dataset, max(k, 0))
            destination = write_model(
                results, save_directory, os.path.basename(dataset_path)
            )
        except PipelineError as error:
            _abort(str(error))
        print(f"Saved student_search_results to {destination}")

    def answer(
        self,
        query: str = "",
        k: int = DEFAULT_K,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> None:
        """Answer a single query from the retrieved context.

        Args:
            query: The natural-language question.
            k: Maximum number of sources to retrieve.
            index_directory: Directory holding the persisted index.
            model_name: Identifier of the generation model.
            max_new_tokens: Upper bound on the generated answer length.
            max_sources: Maximum number of excerpts handed to the model.
        """
        query = str(query).strip()
        if not query:
            _abort("a non-empty query is required")
        sources = _retriever(index_directory).search(query, max(k, 0))
        _print_sources(sources)
        generator = AnswerGenerator(model_name, max_new_tokens, max_sources=max_sources)
        try:
            print(f"\n{generator.answer(query, sources)}")
        except GenerationError as error:
            _abort(str(error))

    def answer_dataset(
        self,
        student_search_results_path: str = "",
        save_directory: str = DEFAULT_ANSWER_DIRECTORY,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_sources: int = DEFAULT_MAX_SOURCES,
        limit: Optional[int] = None,
    ) -> None:
        """Generate an answer for every question of a search results file.

        Args:
            student_search_results_path: Path of the search results JSON file.
            save_directory: Directory the answers are written to.
            model_name: Identifier of the generation model.
            max_new_tokens: Upper bound on the generated answer length.
            max_sources: Maximum number of excerpts handed to the model.
            limit: Stop after this many questions, for quick trials.
        """
        if not student_search_results_path:
            _abort("--student_search_results_path is required")
        try:
            results = load_student_search_results(student_search_results_path)
        except PipelineError as error:
            _abort(str(error))
        questions = results.search_results[:limit] if limit else results.search_results
        print(f"Loaded {len(questions)} questions")
        generator = AnswerGenerator(model_name, max_new_tokens, max_sources=max_sources)
        answers: List[MinimalAnswer] = []
        try:
            for result in tqdm(questions, desc="Answering", unit="question"):
                answer = generator.answer(result.question, result.retrieved_sources)
                answers.append(
                    MinimalAnswer(
                        question_id=result.question_id,
                        question=result.question,
                        retrieved_sources=result.retrieved_sources,
                        answer=answer,
                    )
                )
            destination = write_model(
                StudentSearchResultsAndAnswer(search_results=answers, k=results.k),
                save_directory,
                os.path.basename(student_search_results_path),
            )
        except GenerationError as error:
            _abort(str(error))
        except PipelineError as error:
            _abort(str(error))
        print(f"Saved student_search_results_and_answer to {destination}")

    def evaluate(
        self,
        student_search_results_path: str = "",
        dataset_path: str = "",
        k: int = DEFAULT_K,
    ) -> None:
        """Score retrieval results against a ground-truth dataset.

        Args:
            student_search_results_path: Path of the search results JSON file.
            dataset_path: Path of the ground-truth dataset JSON file.
            k: Largest cutoff to report.
        """
        if not student_search_results_path or not dataset_path:
            _abort("--student_search_results_path and --dataset_path are required")
        try:
            results = load_student_search_results(student_search_results_path)
            dataset = load_rag_dataset(dataset_path)
        except PipelineError as error:
            _abort(str(error))
        cutoffs = sorted({1, 3, 5, max(k, 1)})
        scores = recall_at_k(results, dataset, cutoffs)
        print(format_report(scores, len(results.search_results)))

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        index_directory: str = DEFAULT_INDEX_DIRECTORY,
    ) -> None:
        """Expose search and answering over a local HTTP API.

        Args:
            host: Interface the server binds to.
            port: Port the server listens on.
            index_directory: Directory holding the persisted index.
        """
        try:
            import uvicorn

            from .api import create_app

            uvicorn.run(create_app(index_directory), host=host, port=port)
        except (ImportError, IndexError_, OSError) as error:
            _abort(f"cannot start the API server: {error}")


def main() -> None:
    """Run the CLI, mapping interruptions to a clean exit."""
    try:
        fire.Fire(RagCli(), name="python -m src")
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except BrokenPipeError:
        raise SystemExit(0) from None


if __name__ == "__main__":
    main()
