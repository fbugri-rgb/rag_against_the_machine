"""Local recall@k scoring, used to iterate without the official grader."""

from typing import Dict, List, Sequence

from .models import AnsweredQuestion, MinimalSource, RagDataset, StudentSearchResults

IOU_THRESHOLD = 0.05
DEFAULT_CUTOFFS = (1, 3, 5, 10)


def intersection_over_union(left: MinimalSource, right: MinimalSource) -> float:
    """Return the overlap ratio of two character spans of the same file.

    Args:
        left: First source location.
        right: Second source location.

    Returns:
        The intersection over union of the two spans, zero when the spans lie
        in different files or do not overlap.
    """
    if left.file_path != right.file_path:
        return 0.0
    start = max(left.first_character_index, right.first_character_index)
    end = min(left.last_character_index, right.last_character_index)
    intersection = end - start
    if intersection <= 0:
        return 0.0
    union = (
        max(left.last_character_index, right.last_character_index)
        - min(left.first_character_index, right.first_character_index)
    )
    return intersection / union if union > 0 else 0.0


def _is_found(reference: MinimalSource, retrieved: Sequence[MinimalSource]) -> bool:
    """Tell whether a reference source is covered by retrieved sources.

    Args:
        reference: The ground-truth source location.
        retrieved: The retrieved source locations to search.

    Returns:
        True when one retrieved source overlaps the reference enough.
    """
    return any(
        intersection_over_union(reference, candidate) > IOU_THRESHOLD
        for candidate in retrieved
    )


def recall_at_k(
    results: StudentSearchResults,
    dataset: RagDataset,
    cutoffs: Sequence[int] = DEFAULT_CUTOFFS,
) -> Dict[int, float]:
    """Compute recall at several cutoffs against a ground-truth dataset.

    For a question, recall is the share of its reference sources covered by
    the top-k retrieved sources. The reported value is the mean over every
    question that carries reference sources.

    Args:
        results: The retrieval output to score.
        dataset: The ground-truth dataset.
        cutoffs: The values of k to report.

    Returns:
        Mean recall per cutoff; zeros when the dataset has no answered
        question.
    """
    retrieved_by_id = {
        result.question_id: result.retrieved_sources
        for result in results.search_results
    }
    references = [
        question
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion) and question.sources
    ]
    scores: Dict[int, float] = {cutoff: 0.0 for cutoff in cutoffs}
    if not references:
        return scores
    for question in references:
        retrieved = retrieved_by_id.get(question.question_id, [])
        for cutoff in cutoffs:
            top = retrieved[:cutoff]
            found = sum(1 for source in question.sources if _is_found(source, top))
            scores[cutoff] += found / len(question.sources)
    return {cutoff: value / len(references) for cutoff, value in scores.items()}


def format_report(scores: Dict[int, float], questions: int) -> str:
    """Render a recall report as printable lines.

    Args:
        scores: Mean recall per cutoff.
        questions: Number of questions the scores were computed over.

    Returns:
        The formatted report.
    """
    lines: List[str] = [
        "Evaluation Results",
        "=" * 40,
        f"Questions evaluated: {questions}",
    ]
    for cutoff in sorted(scores):
        value = scores[cutoff]
        lines.append(f"Recall@{cutoff}: {value:.3f} ({value * 100:.1f}%)")
    return "\n".join(lines)
