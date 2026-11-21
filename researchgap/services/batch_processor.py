"""Handle up to five PDF files, including failures and duplicates."""

from dataclasses import dataclass
import hashlib

from researchgap.core.config import MAX_PAPERS, ProcessingSettings
from researchgap.core.schemas import BatchResult, ProcessingFailure
from researchgap.tools.pdf_processor import PDFProcessingError, process_pdf, safe_filename


@dataclass(frozen=True)
class PaperUpload:
    filename: str
    content: bytes


def process_batch(
    uploads: list[PaperUpload],
    settings: ProcessingSettings | None = None,
) -> BatchResult:
    """Continue processing valid documents when individual uploads fail."""
    result = BatchResult(requested_count=len(uploads))
    if len(uploads) > MAX_PAPERS:
        raise ValueError(f"Upload at most {MAX_PAPERS} PDFs at a time.")

    seen_hashes: set[str] = set()
    for upload in uploads:
        name = safe_filename(upload.filename)
        digest = hashlib.sha256(upload.content).hexdigest()
        if digest in seen_hashes:
            result.failures.append(
                ProcessingFailure(
                    filename=name,
                    code="DUPLICATE_PDF",
                    message="The same PDF content was already uploaded in this batch.",
                )
            )
            continue
        seen_hashes.add(digest)
        try:
            paper = process_pdf(name, upload.content, settings)
            result.papers.append(paper)
        except PDFProcessingError as exc:
            result.failures.append(
                ProcessingFailure(filename=name, code=exc.code, message=str(exc))
            )
    return result
