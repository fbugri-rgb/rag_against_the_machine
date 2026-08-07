"""Tests for the two chunking strategies."""

from pathlib import Path
from typing import List

import pytest

from src.chunking import Chunk, chunk_file, chunk_python, chunk_text

PYTHON_SAMPLE = '''"""Module docstring."""

import os


CONSTANT = 1


class Engine:
    """A class."""

    def start(self) -> None:
        """Start it."""
        print(os.getpid())

    def stop(self) -> None:
        """Stop it."""
        return None


def helper(value: int) -> int:
    """Double a value."""
    return value * 2
'''

MARKDOWN_SAMPLE = """# Title

Intro paragraph.

## Section one

Body of section one.

### Nested

More text here.

## Section two

Final body.
"""


def _assert_valid(chunks: List[Chunk], text: str, max_size: int) -> None:
    """Check the invariants every chunk must satisfy."""
    assert chunks, "chunking produced nothing"
    for chunk in chunks:
        span = chunk.last_character_index - chunk.first_character_index
        assert 0 < span <= max_size
        assert chunk.first_character_index >= 0
        assert chunk.last_character_index <= len(text)
        start, end = chunk.first_character_index, chunk.last_character_index
        assert text[start:end] == chunk.body
    starts = [chunk.first_character_index for chunk in chunks]
    assert starts == sorted(starts)


@pytest.mark.parametrize("max_size", [80, 200, 2000])
def test_python_chunks_slice_back_out_of_the_source(max_size: int) -> None:
    """Every Python chunk must be a verbatim slice within the size cap."""
    chunks = chunk_python("sample.py", PYTHON_SAMPLE, max_size)
    _assert_valid(chunks, PYTHON_SAMPLE, max_size)


@pytest.mark.parametrize("max_size", [60, 200, 2000])
def test_markdown_chunks_slice_back_out_of_the_document(max_size: int) -> None:
    """Every Markdown chunk must be a verbatim slice within the size cap."""
    chunks = chunk_text("sample.md", MARKDOWN_SAMPLE, max_size)
    _assert_valid(chunks, MARKDOWN_SAMPLE, max_size)


def test_python_chunks_cover_every_definition() -> None:
    """Small definitions must stay grouped, large ones must be split apart."""
    chunks = chunk_python("sample.py", PYTHON_SAMPLE, 120)
    covered = "".join(chunk.body for chunk in chunks)
    for symbol in ("CONSTANT", "class Engine", "def start", "def stop", "def helper"):
        assert symbol in covered


def test_markdown_context_carries_the_heading_trail() -> None:
    """A nested section must remember the headings enclosing it."""
    chunks = chunk_text("sample.md", MARKDOWN_SAMPLE, 60)
    nested = [chunk for chunk in chunks if "More text here" in chunk.body]
    assert nested
    assert "Nested" in nested[0].context


def test_unparsable_python_falls_back_to_text_chunking() -> None:
    """A syntax error must not remove a file from the index."""
    broken = "def broken(:\n    this is not python\n" * 20
    chunks = chunk_python("broken.py", broken, 200)
    _assert_valid(chunks, broken, 200)


def test_whitespace_only_file_yields_no_chunk() -> None:
    """A file holding only whitespace has nothing worth indexing."""
    assert chunk_file("blank.md", ".md", "   \n\n  \t\n", 2000) == []


def test_real_corpus_file_round_trips(tmp_path: Path) -> None:
    """Chunks of a non-ASCII file must still slice back exactly."""
    text = "# Título\n\nParágrafo com acentuação.\n\n## Fim\n\nTexto final.\n"
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    _assert_valid(chunk_file(str(path), ".md", text, 40), text, 40)
