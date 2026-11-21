"""Extract page-linked text from untrusted, user-uploaded PDF bytes."""

import hashlib
from pathlib import Path
import re

import pymupdf

from researchgap.core.config import (
    MAX_CHARACTERS_PER_PAPER,
    MAX_PAGES_PER_PAPER,
    MAX_PDF_BYTES,
    ProcessingSettings,
)
from researchgap.core.schemas import PageText, ProcessedPaper
from researchgap.tools.chunker import create_chunks


class PDFProcessingError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def is_author_or_metadata_text(text: str) -> bool:
    """Detect whether a line or span represents author names, affiliations, or preprint metadata."""
    if not text:
        return False
    lower = text.lower().strip()

    # Author footnote/contribution symbols
    if any(sym in text for sym in ("†", "‡", "§", "¶")):
        return True
    if "*" in text and any(c.isupper() for c in text):
        return True

    # Common preprint/header keywords
    if any(lower.startswith(prefix) for prefix in ("arxiv:", "preprint", "under review", "in proceedings")):
        return True

    # Affiliations and institutions
    affiliation_keywords = (
        "department", "dept.", "university", "institute", "laboratory", "labs",
        "college", "school", "center", "centre", "corporation", "inc.",
        "@cs.", "@", "email:", "e-mail:", "equal contribution", "correspondence to",
        "allen institute", "ibm research", "google research", "meta ai",
    )
    if any(kw in lower for kw in affiliation_keywords):
        return True

    # Comma-separated author list pattern (e.g. "Akari Asai, Zeqiu Wu, Yizhong Wang")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        title_indicators = {
            "learning", "retrieve", "generate", "critique", "using", "towards",
            "based", "approach", "method", "framework", "system", "neural",
            "language", "model", "models", "evaluation", "benchmark", "analysis",
            "survey", "review", "improving", "scaling", "investigating", "generation",
            "augmented", "retrieval", "self-rag", "crag", "rag", "evaluator",
        }
        words_in_text = set(re.findall(r"\b[a-zA-Z-]+\b", lower))
        if words_in_text.intersection(title_indicators):
            return False

        name_like = sum(
            1 for p in parts
            if 1 <= len(p.split()) <= 4 and any(w and w[0].isupper() for w in p.split())
        )
        if name_like >= 2 and name_like >= len(parts) * 0.7:
            return True

    return False


def strip_authors_from_title(title: str) -> str:
    """Cleanly separate research paper title from appended author names and symbols."""
    clean = title.strip()
    if not clean:
        return ""

    # Truncate at author symbols
    for sym in ("†", "‡", "§", "¶"):
        if sym in clean:
            idx = clean.find(sym)
            # Find the start of the word or name before the symbol
            prefix = clean[:idx].strip()
            # If the last word before the symbol looks like an author name, truncate before it
            words = prefix.split()
            if words and len(words) > 3:
                last_word = words[-1]
                # If last word is capitalized name-like (not an acronym)
                if last_word[0].isupper() and last_word.upper() not in ("RAG", "NLP", "CRAG", "LLM", "AI"):
                    prefix = " ".join(words[:-1])
            clean = prefix

    # Truncate before common author affiliations or emails
    lower = clean.lower()
    for marker in ("\nakari", " akari asai", " university of", " ibm research", " allen institute", " arxiv:"):
        idx = lower.find(marker)
        if idx != -1:
            clean = clean[:idx].strip()
            lower = clean.lower()

    return clean.strip()


def format_title_casing(title: str) -> str:
    """Format ALL CAPS titles to clean Title Case while preserving acronyms."""
    clean = strip_authors_from_title(title)
    if not clean.isupper():
        return clean
    acronyms = {
        "RAG", "NLP", "CRAG", "LLM", "LLMS", "QA", "BERT", "GPT", "AI", "ML",
        "IR", "DPR", "KILT",
    }
    minor_words = {
        "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to",
        "from", "by", "with", "in", "of", "through", "as",
    }
    words = clean.split()
    formatted = []
    for word_idx, w in enumerate(words):
        parts = w.split("-")
        clean_parts = []
        for part_idx, p in enumerate(parts):
            trailing_punct = ""
            core = p
            while core and core[-1] in (":", ",", ".", ";", "!", "?", ")", "]"):
                trailing_punct = core[-1] + trailing_punct
                core = core[:-1]
            if core.upper() in acronyms:
                clean_parts.append(core.upper() + trailing_punct)
            elif core.upper() == "SELF":
                clean_parts.append("Self" + trailing_punct)
            elif (word_idx > 0 or part_idx > 0) and core.lower() in minor_words:
                clean_parts.append(core.lower() + trailing_punct)
            else:
                clean_parts.append(core.capitalize() + trailing_punct)
        formatted.append("-".join(clean_parts))
    return " ".join(formatted)


def extract_title_from_pdf(document: pymupdf.Document) -> str | None:
    """Extract complete research paper title using metadata and first-page font hierarchy."""
    meta_title = (document.metadata or {}).get("title") or ""
    meta_title = strip_authors_from_title(meta_title)
    if (
        meta_title
        and len(meta_title) >= 10
        and len(meta_title.split()) >= 3
        and not meta_title.lower().startswith("untitled")
        and not meta_title.lower().endswith(".pdf")
        and not meta_title.lower().endswith(".tex")
        and not (meta_title.lower().startswith("paper") and "_" in meta_title.lower())
        and not is_author_or_metadata_text(meta_title)
    ):
        return format_title_casing(meta_title)

    if document.page_count < 1:
        return None
    page = document[0]
    page_dict = page.get_text("dict")
    h = page.rect.height

    candidate_spans = []
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        for line in b.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text or len(text) < 2:
                    continue
                bbox = span.get("bbox", (0, 0, 0, 0))
                if bbox[1] < h * 0.45:
                    size = round(span.get("size", 0.0), 1)
                    candidate_spans.append((size, bbox[1], bbox[0], text))

    if not candidate_spans:
        return meta_title or None

    clean_spans = [
        s for s in candidate_spans
        if not is_author_or_metadata_text(s[3])
    ]
    if not clean_spans:
        clean_spans = candidate_spans

    max_size = max(s[0] for s in clean_spans)
    if max_size >= 11.0:
        title_spans = []
        # Sort by vertical position first
        sorted_clean = sorted(clean_spans, key=lambda s: (s[1], s[2]))
        for s in sorted_clean:
            if s[0] >= max_size - 0.8:
                if is_author_or_metadata_text(s[3]):
                    break
                title_spans.append(s)
            elif title_spans:
                # If we've already started the title and font dropped below threshold, check if next line is author
                if is_author_or_metadata_text(s[3]):
                    break

        if title_spans:
            title_text = " ".join(s[3] for s in title_spans).strip()
            title_text = strip_authors_from_title(" ".join(title_text.split()))
            if len(title_text.split()) >= 2 and len(title_text) >= 10:
                return format_title_casing(title_text)

    return meta_title or None


def extract_page_text_layout_aware(page: pymupdf.Page) -> str:
    """Extract text respecting two-column and mixed-layout academic paper structures."""
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
    if not text_blocks:
        return ""

    page_width = float(page.rect.width)
    mid = page_width / 2.0

    def is_spanning(b) -> bool:
        w = b[2] - b[0]
        return w > 0.58 * page_width or (b[0] < mid - 0.15 * page_width and b[2] > mid + 0.15 * page_width)

    spanning = [b for b in text_blocks if is_spanning(b)]
    non_spanning = [b for b in text_blocks if not is_spanning(b)]

    left_blocks = [b for b in non_spanning if (b[0] + b[2]) / 2.0 < mid]
    right_blocks = [b for b in non_spanning if (b[0] + b[2]) / 2.0 >= mid]

    has_two_cols = False
    if left_blocks and right_blocks:
        for lb in left_blocks:
            for rb in right_blocks:
                if max(lb[1], rb[1]) < min(lb[3], rb[3]) - 6:
                    has_two_cols = True
                    break
            if has_two_cols:
                break

    if not has_two_cols:
        sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))
        return "\n\n".join(b[4].strip() for b in sorted_blocks)

    spanning_sorted = sorted(spanning, key=lambda b: b[1])
    ordered_blocks = []
    assigned_ids = set()

    for sb in spanning_sorted:
        sb_top = sb[1]
        above_left = [b for b in left_blocks if id(b) not in assigned_ids and b[3] <= sb_top + 8]
        above_right = [b for b in right_blocks if id(b) not in assigned_ids and b[3] <= sb_top + 8]

        for b in sorted(above_left, key=lambda x: x[1]):
            ordered_blocks.append(b)
            assigned_ids.add(id(b))
        for b in sorted(above_right, key=lambda x: x[1]):
            ordered_blocks.append(b)
            assigned_ids.add(id(b))

        ordered_blocks.append(sb)

    remaining_left = [b for b in left_blocks if id(b) not in assigned_ids]
    remaining_right = [b for b in right_blocks if id(b) not in assigned_ids]

    for b in sorted(remaining_left, key=lambda x: x[1]):
        ordered_blocks.append(b)
    for b in sorted(remaining_right, key=lambda x: x[1]):
        ordered_blocks.append(b)

    return "\n\n".join(b[4].strip() for b in ordered_blocks)


def safe_filename(filename: str) -> str:

    """Keep only a display-friendly basename; never use it as a disk path."""
    name = filename.replace("\\", "/").split("/")[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:180] or "unnamed.pdf"


def process_pdf(
    filename: str,
    content: bytes,
    settings: ProcessingSettings | None = None,
) -> ProcessedPaper:
    """Process a PDF entirely in memory (no user documents saved to disk)."""
    settings = settings or ProcessingSettings()
    filename = safe_filename(filename)

    if Path(filename).suffix.lower() != ".pdf":
        raise PDFProcessingError("UNSUPPORTED_FORMAT", "Only .pdf files are supported.")
    if not content:
        raise PDFProcessingError("EMPTY_UPLOAD", "The uploaded PDF is empty.")
    if len(content) > MAX_PDF_BYTES:
        raise PDFProcessingError(
            "FILE_TOO_LARGE",
            f"PDF exceeds the {MAX_PDF_BYTES // (1024 * 1024)} MB limit.",
        )
    if b"%PDF-" not in content[:1024]:
        raise PDFProcessingError("INVALID_PDF", "File does not contain a valid PDF header.")

    digest = hashlib.sha256(content).hexdigest()
    document_id = digest[:16]
    warnings: list[str] = []
    pages: list[PageText] = []
    total_chars = 0
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            if document.needs_pass:
                raise PDFProcessingError(
                    "PASSWORD_PROTECTED",
                    "Password-protected PDFs are not supported. Upload an unlocked copy.",
                )
            page_count = document.page_count
            if page_count < 1:
                raise PDFProcessingError("NO_PAGES", "PDF has no pages.")
            if page_count > MAX_PAGES_PER_PAPER:
                raise PDFProcessingError(
                    "TOO_MANY_PAGES",
                    f"PDF exceeds the {MAX_PAGES_PER_PAPER}-page limit.",
                )

            title = extract_title_from_pdf(document)
            for page_index in range(page_count):
                page = document.load_page(page_index)
                extracted = extract_page_text_layout_aware(page)
                text = extracted.replace("\x00", "").replace("\r\n", "\n").strip()
                total_chars += len(text)
                if total_chars > MAX_CHARACTERS_PER_PAPER:
                    raise PDFProcessingError(
                        "TOO_MUCH_TEXT",
                        "Document exceeds the extracted-text safety limit.",
                    )
                if not text:
                    warnings.append(
                        f"Page {page_index + 1} has no selectable text; "
                        "scanned pages may require OCR (not included in Phase 1)."
                    )
                pages.append(
                    PageText(
                        page_number=page_index + 1,
                        text=text,
                        character_count=len(text),
                    )
                )
    except PDFProcessingError:
        raise
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, ValueError, RuntimeError) as exc:
        raise PDFProcessingError(
            "INVALID_PDF", "The PDF is damaged or cannot be parsed."
        ) from exc

    text_page_count = sum(bool(page.text) for page in pages)
    if not text_page_count:
        raise PDFProcessingError(
            "NO_EXTRACTABLE_TEXT",
            "No selectable text was found. This may be a scanned/image-only PDF; "
            "OCR is outside the Phase 1 scope.",
        )

    chunks = create_chunks(document_id, pages, settings)
    if not chunks:
        raise PDFProcessingError("NO_EXTRACTABLE_TEXT", "No usable text chunks were found.")

    return ProcessedPaper(
        document_id=document_id,
        sha256=digest,
        filename=filename,
        file_size_bytes=len(content),
        title=title.strip() if title else None,
        page_count=len(pages),
        text_page_count=text_page_count,
        total_characters=total_chars,
        pages=pages,
        chunks=chunks,
        warnings=warnings,
    )
