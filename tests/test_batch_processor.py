import pytest

from researchgap.core.schemas import BatchResult
from researchgap.services.batch_processor import PaperUpload, process_batch


def test_three_valid_papers_are_ready(three_papers):
    uploads = [PaperUpload(name, data) for name, data in three_papers]
    result = process_batch(uploads)
    assert result.requested_count == 3
    assert result.successful_count == 3
    assert result.ready_for_analysis
    assert result.total_chunks == 3
    assert len({p.document_id for p in result.papers}) == 3
    assert BatchResult.model_validate_json(result.model_dump_json()) == result


def test_partial_success_and_duplicate(pdf_factory):
    valid = pdf_factory(["Valid paper content."])
    result = process_batch([
        PaperUpload("a.pdf", valid),
        PaperUpload("a-copy.pdf", valid),
        PaperUpload("broken.pdf", b"definitely not a pdf"),
    ])
    assert result.successful_count == 1
    assert not result.ready_for_analysis
    assert [f.code for f in result.failures] == ["DUPLICATE_PDF", "INVALID_PDF"]


def test_no_files_is_a_valid_empty_batch():
    result = process_batch([])
    assert result.requested_count == 0
    assert result.total_chunks == 0
    assert not result.ready_for_analysis


def test_limit_five_pdfs(pdf_factory):
    inputs = [PaperUpload(f"p{i}.pdf", pdf_factory([f"Paper content {i}"])) for i in range(6)]
    with pytest.raises(ValueError, match="at most 5"):
        process_batch(inputs)
