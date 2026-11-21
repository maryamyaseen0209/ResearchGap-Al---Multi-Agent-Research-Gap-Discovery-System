"""Headless PDF processing for scripts, local testing, and future agents."""

import argparse
from pathlib import Path

from researchgap.core.config import ProcessingSettings
from researchgap.services.batch_processor import PaperUpload, process_batch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process PDFs for ResearchGap AI.")
    parser.add_argument("pdfs", nargs="+", type=Path, help="PDFs to process (up to five)")
    parser.add_argument("--output", type=Path, default=Path("extracted_papers.json"))
    parser.add_argument("--chunk-size", type=int, default=2_000)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        settings = ProcessingSettings(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
        )
        uploads = [PaperUpload(p.name, p.read_bytes()) for p in args.pdfs]
        batch = process_batch(uploads, settings)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(batch.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"Processed {batch.successful_count}/{batch.requested_count} PDFs; "
        f"{batch.total_chunks} chunks; {len(batch.failures)} failed."
    )
    print(f"Output saved: {args.output}")
    for failure in batch.failures:
        print(f" - {failure.filename}: {failure.code} — {failure.message}")
    return 0 if not batch.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
