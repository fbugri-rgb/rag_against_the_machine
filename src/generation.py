"""Grounded answer generation with a small local causal model."""

from typing import Any, List, Optional, Sequence

from .models import MinimalSource
from .pipeline import read_source_text

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_MAX_CONTEXT_TOKENS = 3072
DEFAULT_MAX_SOURCES = 5

SYSTEM_PROMPT = (
    "You answer questions about the vLLM codebase. Use only the numbered "
    "excerpts provided by the user. Start with a one-sentence direct answer. "
    "Then, if the excerpts state a condition, requirement, limitation or "
    "exception that applies to it (look for words such as only, must, "
    "requires, not supported, unless, at least), add it in one or two "
    "sentences, copying names, flags and version numbers exactly. If the "
    "excerpts do not contain the answer, say so plainly instead of guessing. "
    "Be concise and technical, and never invent file names, flags or APIs "
    "that are absent from the excerpts."
)
ANSWER_INSTRUCTION = (
    "Answer using only the excerpts above. Give the direct answer first. If "
    "the most relevant excerpt contains a sentence with 'only', 'must', "
    "'requires' or 'not supported' about this topic, restate that limitation "
    "in your answer."
)


class GenerationError(RuntimeError):
    """Raised when the generation model cannot be loaded or run."""


def build_context(
    sources: Sequence[MinimalSource], max_sources: int = DEFAULT_MAX_SOURCES
) -> str:
    """Render retrieved sources as a numbered excerpt list.

    Args:
        sources: The retrieved source locations, best first.
        max_sources: Maximum number of excerpts to include.

    Returns:
        The rendered excerpts, empty when nothing readable was retrieved.
    """
    blocks: List[str] = []
    for position, source in enumerate(sources[:max_sources], start=1):
        text = read_source_text(source).strip()
        if not text:
            continue
        blocks.append(f"[{position}] {source.file_path}\n{text}")
    return "\n\n".join(blocks)


class AnswerGenerator:
    """Generates answers grounded in retrieved excerpts.

    The model is loaded on first use so that retrieval-only commands never pay
    for the deep-learning stack.

    Attributes:
        model_name: Identifier of the causal model to load.
        max_new_tokens: Upper bound on the generated answer length.
        max_context_tokens: Token budget granted to the excerpts.
        max_sources: Maximum number of excerpts handed to the model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_sources: int = DEFAULT_MAX_SOURCES,
    ) -> None:
        """Record the generation settings without loading anything yet.

        Args:
            model_name: Identifier of the causal model to load.
            max_new_tokens: Upper bound on the generated answer length.
            max_context_tokens: Token budget granted to the excerpts.
            max_sources: Maximum number of excerpts handed to the model.
        """
        self.model_name = model_name
        self.max_new_tokens = max(1, max_new_tokens)
        self.max_context_tokens = max(1, max_context_tokens)
        self.max_sources = max(1, max_sources)
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

    def _load(self) -> None:
        """Load the tokenizer and the model once.

        Raises:
            GenerationError: When the model cannot be loaded.
        """
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model: Any = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.float32
            )
            model.to("cuda" if torch.cuda.is_available() else "cpu")
            self._model = model.eval()
        except Exception as error:  # noqa: BLE001 - surfaced as a clean CLI error
            raise GenerationError(
                f"cannot load generation model '{self.model_name}': {error}"
            ) from error

    def _truncate(self, context: str) -> str:
        """Trim the excerpts to the configured token budget.

        Args:
            context: The rendered excerpts.

        Returns:
            The excerpts, shortened when over budget.
        """
        assert self._tokenizer is not None
        token_ids = self._tokenizer(context, add_special_tokens=False)["input_ids"]
        if len(token_ids) <= self.max_context_tokens:
            return context
        kept = token_ids[: self.max_context_tokens]
        return str(self._tokenizer.decode(kept, skip_special_tokens=True))

    def answer(self, question: str, sources: Sequence[MinimalSource]) -> str:
        """Answer a question from its retrieved sources.

        Args:
            question: The question to answer.
            sources: The retrieved source locations, best first.

        Returns:
            The generated answer.

        Raises:
            GenerationError: When the model cannot be loaded or run.
        """
        context = build_context(sources, self.max_sources)
        if not context:
            return "No relevant source was retrieved for this question."
        self._load()
        assert self._tokenizer is not None and self._model is not None
        # The question is stated before and after the excerpts: a small model
        # reads them with the question in mind and still sees it last.
        prompt = (
            f"Question: {question}\n\n"
            f"Excerpts from the vLLM repository:\n\n{self._truncate(context)}\n\n"
            f"Question: {question}\n\n"
            f"{ANSWER_INSTRUCTION}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            import torch

            text = str(
                self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
            inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
            with torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            answer = str(self._tokenizer.decode(new_tokens, skip_special_tokens=True))
        except Exception as error:  # noqa: BLE001 - surfaced as a clean CLI error
            raise GenerationError(f"generation failed: {error}") from error
        return _clean(answer)


def _clean(answer: str) -> str:
    """Strip reasoning markers and whitespace from a raw completion.

    Args:
        answer: The decoded completion.

    Returns:
        The cleaned answer text.
    """
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]
    return answer.strip() or "No answer could be generated from the retrieved sources."
