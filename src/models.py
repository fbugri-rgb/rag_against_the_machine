"""Pydantic data models exchanged between the RAG pipeline stages."""

import uuid
from typing import List, Union

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A single source location inside the ingested corpus.

    Attributes:
        file_path: Corpus-relative path, written exactly as ingested.
        first_character_index: Start offset of the span, inclusive.
        last_character_index: End offset of the span, exclusive.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question with no reference answer attached.

    Attributes:
        question_id: Stable identifier, generated when absent.
        question: Natural-language question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question shipped with its ground-truth sources and answer.

    Attributes:
        sources: Reference source locations supporting the answer.
        answer: Reference answer text.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not.

    Attributes:
        rag_questions: The questions the dataset holds.
    """

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for one question.

    Attributes:
        question_id: Identifier of the question searched.
        question: The question text.
        retrieved_sources: Ranked source locations, best first.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources for one question plus the generated answer.

    Attributes:
        answer: Answer generated from the retrieved sources.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Output of a batch search over a dataset.

    Attributes:
        search_results: Per-question retrieval results.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of a batch answer generation over a dataset.

    Attributes:
        search_results: Per-question retrieval results with answers.
        k: Number of sources requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int
