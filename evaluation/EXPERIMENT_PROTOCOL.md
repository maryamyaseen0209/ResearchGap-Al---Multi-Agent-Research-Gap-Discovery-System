# ResearchGap AI — Real Evaluation Protocol

**Status:** Protocol only. No real research accuracy or superiority is claimed until a human-reviewed experiment is conducted.

## Research question

Given identical Phase 2 facts extracted from 3–5 papers, how does a **single-pass gap-discovery LLM** compare with a **comparison → gap discovery → external literature verification multi-agent pipeline** on evidence grounding, usefulness, errors, speed, and cost?

This is a *downstream architecture comparison* rather than a fair comparison of one agent versus the entire research pipeline. Both systems inherit the same initial Paper Analyzer, whose errors may bias either result. The verifier has additional OpenAlex access; this is a difference to disclose, not to conceal.

## Design

1. Choose **3–5 sets** of 3–5 papers on separate research topics. Record paper title, DOI, publication year, and license. Do not use the three bundled fictional sample PDFs for scientific scoring.
2. Fix one Groq model, temperature, prompt and source budgets. Preserve original PDF filenames, page text, and Phase 2 outputs. Record model version and date, and pin dependency versions.
3. Run Phase 2 **once** for each set; save the structured `paper_analyses.json`. Use those exact same validated facts to generate the single-pass baseline and full multi-agent result. If the workflow creates its own Phase 2 output, confirm the same facts are used, or evaluate that run separately as a full-pipeline comparison.
4. Keep every candidate result (including missing citations and failed outputs). Do not silently drop hard cases. Record the raw error and truncated-context warnings.
5. Two reviewers should ideally assess each candidate **blind to the system name**. Mark whether the proposed gap is directly supported by its cited excerpts (`yes`/`no`); assess usefulness and novelty **separately**, after independent full-text review. Resolve disagreements with documented adjudication.
6. Use held-out paper sets for qualitative error analysis. Report counts and, when possible, confidence intervals; do not call a 10–15 gap experiment conclusive.

## Automatic metrics (software checks)

- Citation integrity = valid exact source-PDF page quotations / all cited fact IDs, with missing IDs counted as invalid.
- Cross-paper citation rate = fraction of candidate gaps citing facts from at least two distinct source papers.
- Invalid-reference gaps = candidates with any missing/altered page quote.
- Traceable limitations = candidates referencing at least one explicitly extracted `limitation` fact.
- LLM calls, failed calls, measured wall time, and OpenAlex search calls (instrumented; **not** equivalent to dollar cost).
- API provider token usage and *estimated* cost only when the provider reports tokens and you supply valid model-specific input/output prices. Never assume unspecified pricing.

## Human-required metrics

- **Supported by evidence** — does the *meaning* of the full research claim logically follow from its citations? Can be scored using `human_review_template.csv` (`yes` or `no`). Blank is missing, not a negative example.
- **Scientifically novel** — must be checked by real independent literature search and full-text review, and cannot be decided from OpenAlex top results or API abstracts alone.
- **Importance and feasibility** — independent reviewer ratings, not equal to the model's self-rating.

## Reporting

Report baseline and multi-agent performance on exactly the same topic sets; give both raw totals and rates. Show failure examples with page references, ambiguous annotations and missing data. A fully valid source quote is a **provenance check**, not scientific correctness. For `no_direct_match_found`, explicitly state the limited search scope.

### Human review form

The evaluation CLI writes `human_review_template.csv`. For each row fill `supported_by_evidence` with `yes`, `no` or leave blank; `reviewer_notes` may contain the justification. Avoid including personal details or copyright-protected full paper text in a public repository.

## Reproducibility commands

```bash
python -m researchgap.evaluation.baseline_cli --analyses researchgap_paper_analyses.json --output baseline_results.json
python -m researchgap.evaluation.cli --batch researchgap_extracted_papers.json --baseline baseline_results.json --workflow researchgap_workflow_report.json --output-dir evaluation_output/real
# Then label evaluation_output/real/human_review_template.csv and rerun with:
python -m researchgap.evaluation.cli --batch researchgap_extracted_papers.json --baseline baseline_results.json --workflow researchgap_workflow_report.json --labels evaluation_output/real/human_review_template.csv --output-dir evaluation_output/labeled
```

The workflow CLI also emits `researchgap_workflow_report_telemetry.json`. Baseline CLI emits `baseline_telemetry.json`. A baseline comparison requires that both runs use the **same Phase 2 facts**; confirm before reporting any model comparison.
