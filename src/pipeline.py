"""Dataset-level orchestration and JSON input/output helpers."""

import json
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from .models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from .retrieval import Retriever


class PipelineError(RuntimeError):
    """Raised when an input file is missing, unreadable or malformed."""


def _load_json(path: str) -> object:
    """Read and parse a JSON file.

    Args:
        path: Path of the file to read.

    Returns:
        The parsed JSON payload.

    Raises:
        PipelineError: When the file is missing, unreadable or invalid JSON.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as error:
        raise PipelineError(f"file not found: '{path}'") from error
    except json.JSONDecodeError as error:
        raise PipelineError(f"malformed JSON in '{path}': {error}") from error
    except OSError as error:
        raise PipelineError(f"cannot read '{path}': {error}") from error


def load_rag_dataset(path: str) -> RagDataset:
    """Load a question dataset.

    Args:
        path: Path of the dataset JSON file.

    Returns:
        The parsed dataset.

    Raises:
        PipelineError: When the file cannot be read or does not validate.
    """
    try:
        return RagDataset.model_validate(_load_json(path))
    except ValidationError as error:
        raise PipelineError(f"'{path}' is not a valid RagDataset: {error}") from error


def load_student_search_results(path: str) -> StudentSearchResults:
    """Load a retrieval output file.

    Args:
        path: Path of the search results JSON file.

    Returns:
        The parsed search results.

    Raises:
        PipelineError: When the file cannot be read or does not validate.
    """
    try:
        return StudentSearchResults.model_validate(_load_json(path))
    except ValidationError as error:
        raise PipelineError(
            f"'{path}' is not a valid StudentSearchResults: {error}"
        ) from error


def write_model(model: BaseModel, directory: str, filename: str) -> str:
    """Serialise a model to ``directory/filename``.

    Args:
        model: The pydantic model to write.
        directory: Destination directory, created when missing.
        filename: Name of the destination file.

    Returns:
        The path the model was written to.

    Raises:
        PipelineError: When the destination cannot be written.
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        destination = os.path.join(directory, filename)
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(model.model_dump_json(indent=2))
    except OSError as error:
        raise PipelineError(f"cannot write to '{directory}': {error}") from error
    return destination


def search_dataset(
    retriever: Retriever, dataset: RagDataset, k: int, show_progress: bool = True
) -> StudentSearchResults:
    """Run retrieval over every question of a dataset.

    Args:
        retriever: The retriever to query.
        dataset: The dataset holding the questions.
        k: Maximum number of sources per question.
        show_progress: Whether to display a progress bar.

    Returns:
        The retrieval results for the whole dataset.
    """
    results: List[MinimalSearchResults] = []
    for question in tqdm(
        dataset.rag_questions,
        desc="Searching",
        unit="question",
        disable=not show_progress,
    ):
        results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=retriever.search(question.question, k),
            )
        )
    return StudentSearchResults(search_results=results, k=k)


def read_source_text(source: MinimalSource) -> str:
    """Slice the text a source location points at out of the corpus.

    Args:
        source: The source location to read.

    Returns:
        The referenced text, or an empty string when the file is unreadable.
    """
    try:
        with open(
            source.file_path, encoding="utf-8", errors="replace", newline=""
        ) as handle:
            text = handle.read()
    except OSError:
        return ""
    start, end = source.first_character_index, source.last_character_index
    return text[start:end]
