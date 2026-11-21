import pytest
from pydantic import ValidationError

from researchgap.core.config import ProcessingSettings
from researchgap.core.schemas import PageText
from researchgap.tools.chunker import create_chunks


def test_page_boundaries_and_exact_offsets():
    pages = [
        PageText(page_number=1, text="Paragraph one. " * 85, character_count=15 * 85),
        PageText(page_number=2, text="Different page. " * 33, character_count=16 * 33),
    ]
    settings = ProcessingSettings(chunk_size=220, chunk_overlap=35)
    chunks = create_chunks("doc123", pages, settings)
    assert len(chunks) > 4
    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert [chunk.chunk_index for chunk in chunks] == list(range(1, len(chunks) + 1))
    for chunk in chunks:
        assert len(chunk.text) <= settings.chunk_size
        assert 0 <= chunk.start_char < chunk.end_char <= len(pages[chunk.page_number - 1].text)
        assert chunk.text == pages[chunk.page_number - 1].text[chunk.start_char:chunk.end_char]


def test_chunk_overlap_preserves_evidence():
    page = PageText(page_number=1, text="abcde " * 100, character_count=600)
    chunks = create_chunks("d1", [page], ProcessingSettings(chunk_size=200, chunk_overlap=25))
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_char == prev.end_char - 25
        assert page.text[nxt.start_char:prev.end_char] == prev.text[-25:]


def test_empty_page_has_no_chunks():
    assert create_chunks("doc", [PageText(page_number=3, text="", character_count=0)]) == []


@pytest.mark.parametrize("size,overlap", [(199, 10), (200, -1), (200, 200), (250, 300)])
def test_invalid_chunk_settings(size, overlap):
    with pytest.raises(ValidationError):
        ProcessingSettings(chunk_size=size, chunk_overlap=overlap)
