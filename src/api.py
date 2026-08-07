"""Local HTTP API exposing the retrieval and answering pipeline."""

from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generation import DEFAULT_MODEL, AnswerGenerator, GenerationError
from .models import MinimalAnswer, MinimalSearchResults, MinimalSource
from .retrieval import load_retriever


class SearchRequest(BaseModel):
    """A search or answer request.

    Attributes:
        query: The natural-language question.
        k: Maximum number of sources to return.
    """

    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=0, le=100)


def create_app(
    index_directory: str = "data/processed", model_name: str = DEFAULT_MODEL
) -> FastAPI:
    """Build the API application around a persisted index.

    The index is loaded once at start-up and the generation model lazily on
    the first answer request, so search stays available even when no model
    weights are present.

    Args:
        index_directory: Directory holding the persisted index.
        model_name: Identifier of the generation model.

    Returns:
        The configured application.

    Raises:
        IndexError_: When no index can be loaded from ``index_directory``.
    """
    retriever, index = load_retriever(index_directory)
    generator = AnswerGenerator(model_name)
    app = FastAPI(title="RAG against the machine", version="1.0.0")

    @app.get("/health")
    def health() -> Dict[str, object]:
        """Report that the service is up and how large its index is."""
        return {
            "status": "ok",
            "chunks": index.chunk_count,
            "files": len(index.file_paths),
            "max_chunk_size": index.max_chunk_size,
        }

    @app.post("/search")
    def search(request: SearchRequest) -> MinimalSearchResults:
        """Return the top-k sources for a query."""
        return MinimalSearchResults(
            question_id="",
            question=request.query,
            retrieved_sources=_search(request),
        )

    @app.post("/answer")
    def answer(request: SearchRequest) -> MinimalAnswer:
        """Return an answer grounded in the top-k sources for a query."""
        sources = _search(request)
        try:
            text = generator.answer(request.query, sources)
        except GenerationError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return MinimalAnswer(
            question_id="",
            question=request.query,
            retrieved_sources=sources,
            answer=text,
        )

    def _search(request: SearchRequest) -> List[MinimalSource]:
        """Run retrieval for a request."""
        return retriever.search(request.query, request.k)

    return app
