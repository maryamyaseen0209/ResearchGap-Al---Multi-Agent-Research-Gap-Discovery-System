"""In-memory PDF fixtures. The test suite never needs the user's actual PDFs."""

import pymupdf
import pytest


def make_pdf(text_by_page: list[str], *, title: str = "Test Research Paper") -> bytes:
    doc = pymupdf.open()
    doc.set_metadata({"title": title})
    for content in text_by_page:
        page = doc.new_page()
        if content:
            # Use a text box to accommodate multiple words and paragraphs.
            page.insert_textbox(
                pymupdf.Rect(35, 35, 565, 800),
                content,
                fontsize=11,
                lineheight=1.15,
            )
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def pdf_factory():
    return make_pdf


@pytest.fixture
def three_papers(pdf_factory):
    return [
        ("a.pdf", pdf_factory(["Paper A focuses on methods and evaluation."])),
        ("b.pdf", pdf_factory(["Paper B compares datasets and experiments."])),
        ("c.pdf", pdf_factory(["Paper C discusses research limitations."])),
    ]
