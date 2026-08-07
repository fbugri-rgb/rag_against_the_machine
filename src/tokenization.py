"""Identifier-aware tokenizer shared by the indexer and the retriever.

Questions and source code rarely spell things the same way: a question says
"fused batched MoE layer" where the code says ``fused_batched_moe``. The
tokenizer therefore emits both the whole identifier and its sub-words, so a
paraphrase and a verbatim quote can both match the same chunk.
"""

import re
from typing import Iterator, List

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
CAMEL_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+")

STOPWORDS = frozenset(
    """
    a an the this that these those there here and or but if then else when
    while for of in on at to from by with without into onto over under about
    as is are was were be been being am do does did done doing have has had
    having can could shall should will would may might must not no nor so
    such than too very s t just also only own same how what which who whom
    whose why where whether it its it's i you your yours he she they them
    their our we us me my mine any all both each few more most other some
    up down out off again further once between against during before after
    above below through because until unless upon within along across per
    does doesn't don't isn't aren't wasn't weren't use used using does
    """.split()
)


def _split_identifier(token: str) -> List[str]:
    """Split an identifier into its snake_case and camelCase sub-words.

    Args:
        token: The raw identifier to split.

    Returns:
        The sub-words, or an empty list when the identifier is atomic.
    """
    parts: List[str] = []
    for piece in token.split("_"):
        if piece:
            parts.extend(CAMEL_RE.findall(piece))
    if len(parts) <= 1:
        return []
    return parts


def normalize(word: str) -> str:
    """Fold a word to the form stored in the index.

    Applies a deliberately shallow plural rule so that "endpoints" and
    "endpoint" collide without the over-stemming a full stemmer would bring to
    identifier-heavy text.

    Args:
        word: The word to fold.

    Returns:
        The normalized form.
    """
    word = word.lower()
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def iter_tokens(text: str) -> Iterator[str]:
    """Yield the index terms found in ``text``.

    Args:
        text: The text to tokenize.

    Yields:
        Normalized terms, including sub-words of compound identifiers.
    """
    for match in WORD_RE.finditer(text):
        raw = match.group(0).strip("_")
        if not raw:
            continue
        term = normalize(raw)
        if term and term not in STOPWORDS:
            yield term
        for part in _split_identifier(raw):
            sub = normalize(part)
            if len(sub) > 1 and sub != term and sub not in STOPWORDS:
                yield sub


def tokenize(text: str) -> List[str]:
    """Tokenize ``text`` into a list of index terms.

    Args:
        text: The text to tokenize.

    Returns:
        The terms found, in order of appearance.
    """
    return list(iter_tokens(text))
