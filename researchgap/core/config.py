"""Phase 1 document limits and text-chunking settings."""

from pydantic import BaseModel, Field, model_validator

MIN_PAPERS = 3
MAX_PAPERS = 5
MAX_PDF_SIZE_MB = 20
MAX_PDF_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
MAX_PAGES_PER_PAPER = 100
MAX_CHARACTERS_PER_PAPER = 1_000_000


class ProcessingSettings(BaseModel):
    """Validated knobs for deterministic text chunking."""

    chunk_size: int = Field(default=2_000, ge=200, le=20_000)
    chunk_overlap: int = Field(default=200, ge=0)

    @model_validator(mode="after")
    def check_overlap(self) -> "ProcessingSettings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self
