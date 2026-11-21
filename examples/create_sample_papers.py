"""Create three fictional, locally generated research PDFs for smoke testing.

These are NOT published research studies. Their claims and results are invented.
"""

from pathlib import Path

import pymupdf


SAMPLES = [
    (
        "sample_education_rag.pdf",
        "Fictional Study A: Retrieval-Augmented Tutoring",
        [
            "Research objective: Explore retrieval-augmented tutoring for undergraduate students. "
            "Methodology: This fictional demonstration uses a small FAQ corpus and a simulated class. "
            "Dataset: 120 fabricated questions. Finding: The mock system responded more consistently "
            "when relevant examples were supplied. Limitation: No evaluation was conducted in Urdu.",
            "Discussion: For a real study, the investigator would require ethics review, a real dataset, "
            "human evaluation of factuality, and a comparison with an appropriate baseline.",
        ],
    ),
    (
        "sample_multilingual_learning.pdf",
        "Fictional Study B: Multilingual Educational Chatbot",
        [
            "Research objective: Examine language support in education assistants. "
            "Methodology: A hypothetical interface was evaluated with simulated English-language prompts. "
            "Dataset: 80 fabricated prompts. Finding: The fictional model gave inconsistent responses "
            "to code-switched prompts. Limitation: No real-world user study was performed.",
            "Discussion: Code switching and Roman Urdu deserve explicit evaluation, but these toy "
            "examples alone cannot establish that a research gap is globally novel.",
        ],
    ),
    (
        "sample_student_feedback.pdf",
        "Fictional Study C: Automated Student Feedback",
        [
            "Research objective: Study automated qualitative feedback on student writing. "
            "Methodology: Compare two fictional prompt designs using simulated assignments. "
            "Dataset: 60 fabricated writing samples. Finding: The toy examples varied in feedback "
            "specificity. Limitation: No robustness tests or external validation were included.",
            "Discussion: Additional real data, transparent criteria, and human review would be "
            "necessary before deploying an automated feedback system.",
        ],
    ),
]


def main() -> None:
    destination = Path(__file__).parent / "sample_papers"
    destination.mkdir(exist_ok=True)
    for name, title, sections in SAMPLES:
        doc = pymupdf.open()
        doc.set_metadata({"title": title, "author": "ResearchGap AI synthetic fixture"})
        for section in sections:
            page = doc.new_page()
            page.insert_text((50, 70), title, fontsize=15)
            page.insert_textbox(
                pymupdf.Rect(50, 100, 540, 740),
                "SYNTHETIC EXAMPLE — NOT A REAL RESEARCH PAPER\n\n" + section,
                fontsize=11,
            )
        doc.save(destination / name)
        doc.close()
        print(f"Generated: {destination / name}")


if __name__ == "__main__":
    main()
