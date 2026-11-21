"""Typed output contracts that Phase 2 agents can consume directly."""

from pydantic import BaseModel, Field


class PageText(BaseModel):
    page_number: int = Field(ge=1)
    text: str
    character_count: int = Field(ge=0)


class TextChunk(BaseModel):
    """Character offsets index into `PageText.text` (end is exclusive)."""

    chunk_id: str
    document_id: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str


class ProcessedPaper(BaseModel):
    document_id: str
    sha256: str
    filename: str
    file_size_bytes: int = Field(gt=0)
    title: str | None = None
    page_count: int = Field(ge=1)
    text_page_count: int = Field(ge=1)
    total_characters: int = Field(gt=0)
    pages: list[PageText]
    chunks: list[TextChunk]
    warnings: list[str] = Field(default_factory=list)


class ProcessingFailure(BaseModel):
    filename: str
    code: str
    message: str


class BatchResult(BaseModel):
    requested_count: int = Field(ge=0)
    papers: list[ProcessedPaper] = Field(default_factory=list)
    failures: list[ProcessingFailure] = Field(default_factory=list)

    @property
    def successful_count(self) -> int:
        return len(self.papers)

    @property
    def total_chunks(self) -> int:
        return sum(len(paper.chunks) for paper in self.papers)

    @property
    def ready_for_analysis(self) -> bool:
        """Phase 2 will require 3+ distinct, successfully processed PDFs."""
        return self.successful_count >= 3
