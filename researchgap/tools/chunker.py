"""Generate reproducible, page-aligned overlapping text chunks."""

from researchgap.core.config import ProcessingSettings
from researchgap.core.schemas import PageText, TextChunk


def create_chunks(
    document_id: str,
    pages: list[PageText],
    settings: ProcessingSettings | None = None,
) -> list[TextChunk]:
    """Chunk each page separately so every excerpt has an exact source page.

    start_char is zero-based, end_char is exclusive. Each chunk's text is
    byte-for-byte (as a Python string) the matching slice of its page text.
    Sentence/whitespace boundaries are preferred when they don't cause tiny
    chunks; otherwise a fixed-size cut is used.
    """
    settings = settings or ProcessingSettings()
    result: list[TextChunk] = []
    for page in pages:
        text = page.text
        start = 0
        while start < len(text):
            end = min(start + settings.chunk_size, len(text))
            if end < len(text):
                window = text[start:end]
                # Look for a readable boundary in the latter 40% of the window.
                candidates = [
                    window.rfind("\n\n"),
                    window.rfind("\n"),
                    window.rfind(". "),
                    window.rfind(" "),
                ]
                boundary = max(candidates)
                if boundary >= int(settings.chunk_size * 0.6):
                    end = start + boundary + 1
                    # Keep enough new characters to avoid a stalled overlap.
                    if end - start <= settings.chunk_overlap:
                        end = min(start + settings.chunk_size, len(text))

            snippet = text[start:end]
            if snippet.strip():
                index = len(result) + 1
                result.append(
                    TextChunk(
                        chunk_id=f"{document_id}:p{page.page_number}:c{index}",
                        document_id=document_id,
                        page_number=page.page_number,
                        chunk_index=index,
                        start_char=start,
                        end_char=end,
                        text=snippet,
                    )
                )

            if end >= len(text):
                break
            # Always move forward, even if an unusual boundary was selected.
            start = max(start + 1, end - settings.chunk_overlap)

    return result
