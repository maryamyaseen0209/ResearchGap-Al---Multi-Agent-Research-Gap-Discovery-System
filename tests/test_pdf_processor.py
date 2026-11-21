import hashlib

import pymupdf
import pytest

from researchgap.core.config import ProcessingSettings
from researchgap.tools.pdf_processor import PDFProcessingError, process_pdf, safe_filename


def test_page_metadata_and_source_linked_chunks(pdf_factory):
    content = pdf_factory([
        "This is the first page about a research objective and some experiments.",
        "This second page reports a method, dataset, and limitations.",
    ])
    result = process_pdf("papers/demo.pdf", content)
    assert result.filename == "demo.pdf"
    assert result.page_count == 2
    assert result.text_page_count == 2
    assert result.title == "Test Research Paper"
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert result.document_id == result.sha256[:16]
    assert [p.page_number for p in result.pages] == [1, 2]
    assert result.total_characters == sum(p.character_count for p in result.pages)
    assert len(result.chunks) == 2
    for chunk in result.chunks:
        page = result.pages[chunk.page_number - 1]
        assert chunk.text == page.text[chunk.start_char:chunk.end_char]
        assert chunk.chunk_id.startswith(result.document_id + ":p")


def test_windows_filename_sanitized(pdf_factory):
    result = process_pdf("C:\\secret\\project\\paper.PDF", pdf_factory(["Hello world."]))
    assert result.filename == "paper.PDF"
    assert safe_filename("folder/\\bad\x00name.pdf") == "badname.pdf"


@pytest.mark.parametrize(
    "name,data,code",
    [
        ("notes.txt", b"%PDF-1.4 fake", "UNSUPPORTED_FORMAT"),
        ("empty.pdf", b"", "EMPTY_UPLOAD"),
        ("wrong.pdf", b"this is not a pdf", "INVALID_PDF"),
        ("broken.pdf", b"%PDF-1.7\nnot a real body", "INVALID_PDF"),
    ],
)
def test_invalid_input(name, data, code):
    with pytest.raises(PDFProcessingError) as error:
        process_pdf(name, data)
    assert error.value.code == code


def test_blank_page_warns_but_valid_page_works(pdf_factory):
    result = process_pdf("partial.pdf", pdf_factory(["This page has text.", ""]))
    assert result.text_page_count == 1
    assert result.pages[1].text == ""
    assert len(result.warnings) == 1
    assert "Page 2" in result.warnings[0]


def test_fully_image_only_or_blank_document_is_rejected(pdf_factory):
    with pytest.raises(PDFProcessingError) as error:
        process_pdf("scanned.pdf", pdf_factory(["", ""]))
    assert error.value.code == "NO_EXTRACTABLE_TEXT"


def test_password_protected_pdf_is_rejected(pdf_factory):
    original = pymupdf.open(stream=pdf_factory(["Secret paper text."]), filetype="pdf")
    encrypted = original.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="ownerpass",
        user_pw="userpass",
    )
    original.close()
    with pytest.raises(PDFProcessingError) as error:
        process_pdf("encrypted.pdf", encrypted)
    assert error.value.code == "PASSWORD_PROTECTED"


def test_max_size_rejected_before_parsing():
    from researchgap.core.config import MAX_PDF_BYTES

    with pytest.raises(PDFProcessingError) as error:
        process_pdf("oversized.pdf", b"%PDF-1.7" + b" " * MAX_PDF_BYTES)
    assert error.value.code == "FILE_TOO_LARGE"


def test_page_limit_rejected(pdf_factory):
    content = pdf_factory(["a small sentence"] * 101)
    with pytest.raises(PDFProcessingError) as error:
        process_pdf("very_long.pdf", content)
    assert error.value.code == "TOO_MANY_PAGES"


def test_document_json_serialization(pdf_factory):
    result = process_pdf("research.pdf", pdf_factory(["One research question."]))
    assert result.model_validate_json(result.model_dump_json()) == result


def test_custom_chunk_size(pdf_factory):
    result = process_pdf(
        "a.pdf",
        pdf_factory(["Alpha beta gamma delta " * 25]),
        ProcessingSettings(chunk_size=200, chunk_overlap=40),
    )
    assert len(result.chunks) >= 2
    assert all(len(ch.text) <= 200 for ch in result.chunks)


def test_character_safety_limit(pdf_factory, monkeypatch):
    import researchgap.tools.pdf_processor as processor

    monkeypatch.setattr(processor, "MAX_CHARACTERS_PER_PAPER", 20)
    with pytest.raises(PDFProcessingError) as error:
        process_pdf("long.pdf", pdf_factory(["This contains more than twenty characters."]))
    assert error.value.code == "TOO_MUCH_TEXT"
