"""Chunking strategies turning corpus files into indexable character spans.

Two distinct strategies are implemented: an AST-driven chunker for Python
sources, which cuts on statement and definition boundaries, and a structural
chunker for Markdown and plain text, which cuts on headings and paragraphs.
Both emit spans expressed as character offsets into the decoded file, so a
chunk can always be sliced back out of the original file verbatim.
"""

import ast
import re
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(\S.*)$", re.MULTILINE)
PARAGRAPH_RE = re.compile(r"\n[ \t]*\n")

CODE_SUFFIXES = frozenset({".py", ".pyi"})
TEXT_SUFFIXES = frozenset({".md", ".txt", ".rst", ".markdown"})


@dataclass
class Chunk:
    """A retrievable span of a corpus file.

    Attributes:
        file_path: Corpus-relative path of the file the span belongs to.
        first_character_index: Start offset of the span, inclusive.
        last_character_index: End offset of the span, exclusive.
        context: Structural labels (path, headings, symbol names) describing
            where the span sits; indexed with extra weight.
        body: The raw text covered by the span.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    context: str
    body: str


def _line_starts(text: str) -> List[int]:
    """Compute the character offset at which each line of ``text`` starts.

    Args:
        text: The full text to scan.

    Returns:
        Offsets of the first character of every line.
    """
    starts = [0]
    for match in re.finditer("\n", text):
        starts.append(match.end())
    return starts


def _char_offset(text: str, starts: Sequence[int], lineno: int, col: int) -> int:
    """Translate an AST ``(lineno, col_offset)`` pair into a character offset.

    ``col_offset`` is a UTF-8 byte offset, which only differs from a character
    offset on lines holding non-ASCII text.

    Args:
        text: The full source text.
        starts: Offsets of the first character of every line.
        lineno: One-based line number.
        col: Zero-based UTF-8 byte offset within the line.

    Returns:
        The character offset inside ``text``.
    """
    index = min(max(lineno - 1, 0), len(starts) - 1)
    start = starts[index]
    if col <= 0:
        return start
    end = starts[index + 1] if index + 1 < len(starts) else len(text)
    line = text[start:end]
    if not line.isascii():
        col = len(line.encode("utf-8")[:col].decode("utf-8", "ignore"))
    return min(start + col, len(text))


def _emit(
    file_path: str, text: str, start: int, end: int, context: str, max_size: int
) -> Iterator[Chunk]:
    """Yield a chunk for ``text[start:end]``, trimmed and length-capped.

    Args:
        file_path: Corpus-relative path of the file.
        text: The full file text.
        start: Start offset of the span.
        end: End offset of the span.
        context: Structural labels for the span.
        max_size: Hard cap on the span length in characters.

    Yields:
        A single chunk, or nothing when the span holds only whitespace.
    """
    end = min(end, start + max_size, len(text))
    body = text[start:end]
    stripped = body.strip()
    if not stripped:
        return
    start += len(body) - len(body.lstrip())
    end = start + len(stripped)
    yield Chunk(file_path, start, end, context, stripped)


def _split_plain(
    file_path: str, text: str, start: int, end: int, context: str, max_size: int
) -> Iterator[Chunk]:
    """Chunk a span by packing paragraphs, then lines, then hard cuts.

    Args:
        file_path: Corpus-relative path of the file.
        text: The full file text.
        start: Start offset of the span to split.
        end: End offset of the span to split.
        context: Structural labels inherited by every produced chunk.
        max_size: Maximum chunk length in characters.

    Yields:
        Chunks covering the whole span, in order.
    """
    cuts = [start]
    for match in PARAGRAPH_RE.finditer(text, start, end):
        cuts.append(match.end())
    cuts.append(end)

    chunk_start = start
    for index in range(1, len(cuts)):
        boundary = cuts[index]
        if boundary - chunk_start <= max_size:
            continue
        previous = cuts[index - 1]
        if previous > chunk_start:
            yield from _emit(file_path, text, chunk_start, previous, context, max_size)
            chunk_start = previous
        while boundary - chunk_start > max_size:
            cut = text.rfind("\n", chunk_start, chunk_start + max_size)
            if cut <= chunk_start:
                cut = chunk_start + max_size
            yield from _emit(file_path, text, chunk_start, cut, context, max_size)
            chunk_start = cut
    if end > chunk_start:
        yield from _emit(file_path, text, chunk_start, end, context, max_size)


def chunk_text(file_path: str, text: str, max_size: int) -> List[Chunk]:
    """Chunk a Markdown or plain-text document on its heading structure.

    Sections introduced by ATX headings are kept whole when they fit, merged
    with their neighbours when they are small, and split on paragraph
    boundaries when they are too large. Each chunk carries the trail of
    enclosing headings as context.

    Args:
        file_path: Corpus-relative path of the file.
        text: The full file text.
        max_size: Maximum chunk length in characters.

    Returns:
        Chunks covering the whole document, in reading order.
    """
    sections: List[Tuple[int, int, str]] = []
    trail: List[Tuple[int, str]] = []
    matches = list(HEADING_RE.finditer(text))
    cursor = 0
    for index, match in enumerate(matches):
        if match.start() > cursor:
            sections.append((cursor, match.start(), _trail_label(trail)))
        level = len(match.group(1))
        while trail and trail[-1][0] >= level:
            trail.pop()
        trail.append((level, match.group(2).strip()))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.start(), end, _trail_label(trail)))
        cursor = end
    if cursor < len(text):
        sections.append((cursor, len(text), _trail_label(trail)))

    chunks: List[Chunk] = []
    pending: Optional[Tuple[int, int, str]] = None
    for start, end, label in sections:
        if pending is not None and end - pending[0] <= max_size:
            pending = (pending[0], end, pending[2] or label)
            continue
        if pending is not None:
            chunks.extend(_flush_text(file_path, text, pending, max_size))
        pending = (start, end, label)
    if pending is not None:
        chunks.extend(_flush_text(file_path, text, pending, max_size))
    return chunks


def _flush_text(
    file_path: str, text: str, section: Tuple[int, int, str], max_size: int
) -> Iterator[Chunk]:
    """Emit chunks for one accumulated text section.

    Args:
        file_path: Corpus-relative path of the file.
        text: The full file text.
        section: The ``(start, end, heading_trail)`` section to emit.
        max_size: Maximum chunk length in characters.

    Yields:
        Chunks covering the section.
    """
    start, end, label = section
    context = f"{file_path} {label}"
    if end - start <= max_size:
        yield from _emit(file_path, text, start, end, context, max_size)
    else:
        yield from _split_plain(file_path, text, start, end, context, max_size)


def _trail_label(trail: Sequence[Tuple[int, str]]) -> str:
    """Render a heading stack as a single context string.

    Args:
        trail: The stack of ``(level, title)`` headings currently open.

    Returns:
        The heading titles joined from outermost to innermost.
    """
    return " ".join(title for _, title in trail)


def _node_label(node: ast.stmt) -> str:
    """Return the symbol name a Python statement defines, if any.

    Args:
        node: The statement to inspect.

    Returns:
        The defined name, or an empty string for statements that define none.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    return ""


def _node_start(node: ast.stmt, text: str, starts: Sequence[int]) -> int:
    """Return the offset at which a statement starts, decorators included.

    Args:
        node: The statement to locate.
        text: The full source text.
        starts: Offsets of the first character of every line.

    Returns:
        The character offset where the statement begins.
    """
    lineno = node.lineno
    decorators = getattr(node, "decorator_list", [])
    for decorator in decorators:
        lineno = min(lineno, decorator.lineno)
    return starts[min(max(lineno - 1, 0), len(starts) - 1)]


def _chunk_body(
    file_path: str,
    text: str,
    starts: Sequence[int],
    body: Sequence[ast.stmt],
    start: int,
    end: int,
    scope: str,
    max_size: int,
) -> Iterator[Chunk]:
    """Chunk a Python statement list, recursing into oversized definitions.

    Consecutive statements are packed together while they fit. A definition
    larger than ``max_size`` on its own is split across its own body, so
    methods of a long class become separate chunks. The produced spans are
    contiguous, hence comments and blank lines between statements are never
    dropped.

    Args:
        file_path: Corpus-relative path of the file.
        text: The full source text.
        starts: Offsets of the first character of every line.
        body: The statements to chunk.
        start: Start offset of the region the statements cover.
        end: End offset of the region the statements cover.
        scope: Dotted name of the enclosing definition, if any.
        max_size: Maximum chunk length in characters.

    Yields:
        Chunks covering the whole region, in source order.
    """
    spans: List[Tuple[int, int, ast.stmt]] = []
    cursor = start
    for index, node in enumerate(body):
        node_end = (
            _node_start(body[index + 1], text, starts)
            if index + 1 < len(body)
            else end
        )
        spans.append((cursor, max(node_end, cursor), node))
        cursor = max(node_end, cursor)
    if cursor < end and spans:
        spans[-1] = (spans[-1][0], end, spans[-1][2])

    group_start = start
    labels: List[str] = []
    for span_start, span_end, node in spans:
        if span_end - group_start <= max_size:
            label = _node_label(node)
            if label:
                labels.append(label)
            continue
        if span_start > group_start:
            yield from _emit(
                file_path,
                text,
                group_start,
                span_start,
                _scope_label(file_path, scope, labels),
                max_size,
            )
            labels = []
        inner = getattr(node, "body", None)
        if span_end - span_start > max_size and inner:
            name = _node_label(node)
            child_scope = f"{scope}.{name}" if scope and name else (name or scope)
            header_end = _char_offset(text, starts, inner[0].lineno, 0)
            yield from _emit(
                file_path,
                text,
                span_start,
                header_end,
                _scope_label(file_path, child_scope, []),
                max_size,
            )
            yield from _chunk_body(
                file_path,
                text,
                starts,
                inner,
                header_end,
                span_end,
                child_scope,
                max_size,
            )
        else:
            yield from _split_plain(
                file_path,
                text,
                span_start,
                span_end,
                _scope_label(file_path, scope, [_node_label(node)]),
                max_size,
            )
        group_start = span_end
    if end > group_start:
        yield from _emit(
            file_path,
            text,
            group_start,
            end,
            _scope_label(file_path, scope, labels),
            max_size,
        )


def _scope_label(file_path: str, scope: str, labels: Sequence[str]) -> str:
    """Build the context string describing where a code chunk sits.

    Args:
        file_path: Corpus-relative path of the file.
        scope: Dotted name of the enclosing definition, if any.
        labels: Names defined by the statements inside the chunk.

    Returns:
        The context string indexed alongside the chunk body.
    """
    parts = [file_path, scope, *labels]
    return " ".join(part for part in parts if part)


def chunk_python(file_path: str, text: str, max_size: int) -> List[Chunk]:
    """Chunk a Python source file on its definition boundaries.

    Falls back to the plain-text strategy when the file does not parse, so an
    unsupported syntax version never removes a file from the index.

    Args:
        file_path: Corpus-relative path of the file.
        text: The full source text.
        max_size: Maximum chunk length in characters.

    Returns:
        Chunks covering the whole file, in source order.
    """
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return list(_split_plain(file_path, text, 0, len(text), file_path, max_size))
    starts = _line_starts(text)
    return list(
        _chunk_body(file_path, text, starts, tree.body, 0, len(text), "", max_size)
    )


def chunk_file(file_path: str, suffix: str, text: str, max_size: int) -> List[Chunk]:
    """Chunk a file with the strategy matching its type.

    Args:
        file_path: Corpus-relative path of the file.
        suffix: Lower-cased file extension, dot included.
        text: The full file text.
        max_size: Maximum chunk length in characters.

    Returns:
        Chunks covering the whole file.
    """
    if suffix in CODE_SUFFIXES:
        return chunk_python(file_path, text, max_size)
    return chunk_text(file_path, text, max_size)
