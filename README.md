# ResearchGap AI — Phases 1–8 (Final MVP)

**Evidence-Grounded Multi-Agent Research Paper Analysis, Scholarly Verification, LangGraph Orchestration, and Professional Report Studio**

ResearchGap AI accepts **3–5 selectable-text research PDFs**, extracts exact page-linked quotations, analyzes individual papers, compares findings, proposes *candidate* research gaps, and checks related literature through OpenAlex. **Phase 6 provides a single real LangGraph `StateGraph`** that coordinates those four existing agents and can attempt one bounded revision after a challenged gap. The step-by-step Phase 1–5 controls remain available. **Phase 7 adds a redesigned research workspace and traceable reports; Phase 8 adds an Evaluation Lab, a single-pass baseline, automatic citation metrics, human-label templates, test automation, and deployment readiness.**

> **Research integrity:** Exact quotation matching verifies that a source excerpt exists; it does **not** verify the scientific interpretation. OpenAlex searches are bounded and incomplete. No component establishes that a topic is globally novel. All research-gap verdicts require human review.

## Implemented features

| Phase | Component | Outcome |
| --- | --- | --- |
| 1 | PDF extraction | Multi-PDF upload, duplicate/invalid detection, PDF text chunks with exact source-page offsets, JSON export |
| 2 | Paper Analyzer Agent | Groq-powered structured methods, findings, datasets, objectives and limitations; source-quote verification |
| 3 | Paper Comparison Agent | Multi-paper evidence table and comparison insights with fact-ID citations |
| 4 | Gap Discovery Agent | Cautious candidate gaps, research questions and preliminary priorities backed by source facts |
| 5 | Gap Verification Agent | Bounded OpenAlex search and abstract-review evidence; `challenged` / `no_direct_match_found` / `inconclusive` |
| **6** | **LangGraph orchestration** | **Coordinated analyze → compare → discover → verify, conditional routing, one optional challenged-gap revision, in-memory checkpoints, event trace, CLI and one-click Streamlit option** |
| **7** | **Research workspace & Report Studio** | **Responsive dark-themed Streamlit tabs, research progress, report preview, Markdown/HTML/JSON exports, safe rendering** |
| **8** | **Evaluation, CI & deployment** | **Single-pass baseline, citation-integrity metrics, human review CSV, LLM/search-call telemetry, eighth Evaluation tab, offline E2E smoke, GitHub Actions, and Streamlit Cloud checklist** |

## Install

Use Python **3.12**, preferably in a new virtual environment. The `langgraph` dependency is included in `requirements.txt` and installed by `requirements-dev.txt` too.

**Windows PowerShell**

```powershell
cd ResearchGap-AI
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

**macOS/Linux**

```bash
cd ResearchGap-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env` without publishing it:

```dotenv
GROQ_API_KEY=your_private_groq_key
GROQ_MODEL=openai/gpt-oss-120b
OPENALEX_API_KEY=
```

**Important model update:** The old default `llama-3.3-70b-versatile` was deprecated for standard access in August 2026. The new default is `openai/gpt-oss-120b`, with JSON mode and low reasoning effort. Confirm model availability on your Groq plan. Groq key is required for real AI execution. A free OpenAlex API key is optional for small tests but recommended; public quotas and allowed access may change. The Groq model availability can also change.

Launch:

```bash
python -m streamlit run app.py
```

In your browser (normally `http://localhost:8501`):

1. Upload 3–5 unique, readable research PDFs.
2. Click **Process research papers** to extract text **locally**.
3. Click **Run full multi-agent workflow** under Phase 6. This explicitly consents to sending selected PDF excerpts and related scholarly queries to external services.
4. Review the timeline, source-linked candidate gaps, verification verdicts and uncertainty warnings.
5. Click **Download complete LangGraph workflow (JSON)** to save results.

The existing step-by-step buttons (Analyze, Compare, Discover, Verify) can still be used instead of the one-click flow. Neither Groq nor OpenAlex is contacted merely by uploading a file.

## Phase 7 — redesigned interface and reports

The application now has **eight navigable tabs**: **Library**, **Analyze**, **Compare**, **Discover**, **Verify**, **Agent workflow**, **Reports**, and **Evaluation**. A fixed-style sidebar explains the workflow, a stage tracker shows what has been completed, and the responsive visual theme uses a dark navy/violet palette with accessible control focus styles. The design is implemented with native Streamlit controls and custom CSS; no React server, CSS framework or extra paid service is required.

**Recommended workflow:**

1. In **01 · Library**, upload **3–5 distinct, selectable-text PDFs**, then select **Process research papers**. Inspect the source pages and text chunks.
2. In **06 · Agent workflow**, choose **Run full multi-agent workflow**. The action explicitly sends selected source text to Groq and scholarly queries to OpenAlex. Alternatively run the individual phases in tabs 02–05.
3. In **07 · Reports**, optionally edit the report title and author, then choose **Generate research report**. This is an **offline transformation of your existing results**; it does **not** make additional Groq/OpenAlex requests.
4. Download **Markdown (.md)**, **HTML (.html)**, and **JSON (.json)**. HTML is styled for reading and printing. For a PDF, open the HTML file in your browser and use **Print → Save as PDF**.

Every report includes available paper information, page-anchored source quotations, cross-paper comparisons, candidate gaps, research questions, literature verification verdicts and search records, warnings, and the agent execution trace (for one-click runs). Missing phases are marked unavailable rather than invented. **The reports cannot certify research novelty.**

### Offline report demo (no keys or LangGraph installation required)

```bash
python -m examples.demo_phase7_offline
```

This generates a complete **fictional** example report under `examples/sample_reports/`. The bundled HTML/Markdown/JSON samples are labeled as simulations and are **not evidence of real scholarly results**. Open the generated HTML in a browser to preview the print-ready report styling.

### Export a saved Phase 6 workflow from the command line

```bash
python -m researchgap.report_cli researchgap_workflow_report.json \
  --batch researchgap_extracted_papers.json \
  --output-dir my_reports \
  --title "Evidence-Grounded Research Review" \
  --author "Research team"
```

On Windows PowerShell, enter the command on one line or use PowerShell backticks instead of the Unix backslashes. `--batch` is optional. Report export is offline and requires no API key. The command reads a valid Phase 6 workflow JSON and writes Markdown, HTML, and JSON into the specified directory.

**Report integrity:** HTML is escaped before embedding untrusted PDF/model strings; only canonical OpenAlex work URLs are made clickable. The structured JSON preserves the original exact quotations and page offsets. Source quotations verify provenance, **not** scientific accuracy. Re-generate the report after any input, analysis or report-title change to avoid stale exports.

**Testing caveat:** Headless Streamlit widget tests pass, but a live Streamlit browser has not been run in this execution environment because Streamlit is not installed here. Live Groq, OpenAlex and real LangGraph execution also remain unverified in this environment. See `TEST_REPORT.md` for details and commands to run the checks on your machine.

### Phase 6 orchestration graph

```text
User click + processed Phase 1 PDF batch
       |
       v
    Analyze  ----- insufficient successful papers --> Finish
       |
       v
    Compare  ----- invalid or insufficient analysis --> Finish
       |
       v
    Discover ----- no grounded candidate gaps ------> Finish
       |
       v
    Verify   ---- nothing challenged ---------------> Finish
       |
       v
    Revise (at most once)
       |   | no valid new candidates
       |   +-----------------------------------------> Finish
       v
    Reverify revised candidates ---------------------> Finish
```

The state is a typed dictionary with model-validated JSON stage outputs and an append-only event log. The graph contains six named nodes (`analyze`, `compare`, `discover`, `verify`, `revise`, `finish`) and uses LangGraph conditional edges. Individual failures do not erase successfully obtained previous-stage evidence.

#### Revision behavior

Revision is **attempted only if Phase 5 returned at least one `challenged` verdict**. On this single attempt, the discovery agent receives the previously challenged titles as untrusted feedback, must still cite source-validated facts from 2+ original papers, and may not reuse the original challenged title or exact fact combination. Unchallenged candidates are preserved. If it produces no acceptable replacement, the initial `challenged` verdict and references are retained: no evidence is hidden or silently overwritten. If new candidates are accepted, they are verified again against OpenAlex. The graph never loops more than once.

`WorkflowResult` preserves the final paper analyses, comparison, candidate gaps, final literature verification, optional initial verification, revision status, the event trail, and a stable `run_id`. It explicitly includes `novelty_established=false` and `human_review_required=true`.

**Persistence:** By default the workflow uses LangGraph's `InMemorySaver` checkpointer with a fresh thread ID for each invocation; `ResearchWorkflow.resume(run_id)` can continue unfinished work only using the **same live workflow instance**. This is process-local memory, **not** production/database persistence. No background execution or long-running resumable UI job is claimed.

### Phase 6 CLI

First extract PDFs locally (the same command as Phase 1):

```bash
python -m researchgap.cli \
  examples/sample_papers/sample_education_rag.pdf \
  examples/sample_papers/sample_multilingual_learning.pdf \
  examples/sample_papers/sample_student_feedback.pdf \
  --output researchgap_extracted_papers.json
```

Then, with `.env` configured and internet access available, run the real graph:

```bash
python -m researchgap.workflow_cli researchgap_extracted_papers.json \
  --output researchgap_workflow_report.json
```

To disable challenged-gap revision, append `--no-revision`. For PowerShell, run each multi-line command on a single line.

The CLI saves a result even on a partially completed workflow, returning exit code `0` for operationally complete runs and `1` for stopped/partial runs. `complete` means the bounded procedure finished; it does **not** mean the detected research gap is novel.

The older Phase 2–5 CLI modules and their input/output contracts remain usable:

```bash
python -m researchgap.analysis_cli researchgap_extracted_papers.json --output researchgap_paper_analyses.json
python -m researchgap.comparison_cli researchgap_paper_analyses.json --output researchgap_paper_comparison.json
python -m researchgap.gap_cli researchgap_paper_comparison.json --output researchgap_candidate_gaps.json
python -m researchgap.verification_cli researchgap_candidate_gaps.json --output researchgap_literature_verification.json
```


## Phase 8 — Evidence evaluation and deploy readiness

Use the eighth **Evaluation** tab *after* running the full workflow. Click **Run single-pass evaluation baseline** to compare one downstream LLM gap-generation call against the multi-agent workflow, using the same Phase 2 paper facts. The app exports: (1) baseline JSON, (2) citation-integrity metrics JSON, (3) a CSV template for independent reviewers, and (4) optional telemetry when available. The evaluation never runs without an explicit click.

The automatic metrics include exact page-quotation integrity, incorrect/missing source IDs, cross-paper evidence coverage, and source-limitation citations. **These do not establish scientific accuracy or novelty.** When reviewers complete the CSV with `yes`/`no` evidence-support labels, the CLI computes a human-reviewed support rate; without labels, the metric remains null. Read `evaluation/EXPERIMENT_PROTOCOL.md` for the full rubric and threats to validity.

**Offline Phase 1–8 software smoke** (fictional papers, mocked LLM/literature; no Groq/OpenAlex access):

```bash
python -m examples.demo_phase8_offline --output-dir evaluation_output/offline_smoke
python -m researchgap.evaluation.cli --batch evaluation_output/offline_smoke/processed_papers.json --baseline evaluation_output/offline_smoke/baseline_results.json --workflow evaluation_output/offline_smoke/workflow_results.json --output-dir evaluation_output/review
```

The smoke test clearly identifies whether **actual LangGraph** was used. If LangGraph is not installed, only the offline sequential business logic runs; this is not a production graph substitute. To **require real LangGraph**, install the requirements and pass `--require-langgraph`.

To run a **real single-pass Groq baseline** on previously saved Phase 2 facts:

```bash
python -m researchgap.evaluation.baseline_cli --analyses researchgap_paper_analyses.json --output baseline_results.json
```

The workflow CLI saves its own telemetry to `researchgap_workflow_report_telemetry.json`, and the baseline CLI saves `baseline_telemetry.json`. Observed token counts come from the provider when available; cost is **not invented** if pricing or token usage is missing. The app's Evaluation tab displays measured call counts and latency when both sets of results exist.

### CI and deployment

- `.github/workflows/ci.yml` installs actual Streamlit + LangGraph, runs tests, executes the real LangGraph offline E2E workflow, and checks the Streamlit HTTP health endpoint. **The workflow is configured, not yet executed on your GitHub repository.**
- `.streamlit/secrets.toml.example` shows Cloud secret names (actual secrets are ignored by Git).
- `evaluation/DEPLOYMENT.md` has the steps for Streamlit Community Cloud and a deployment security checklist.
- `evaluation/RELEASE_CHECKLIST.md` shows exactly what is tested and what still requires a real browser, your credentials, or independent scientific review.

**Critical:** A public demo with your Groq key configured can incur charges from visitors. Keep access restricted until you have robust authentication, rate limiting, and budget controls. Source PDFs may contain copyright-protected or private text; obtain permission before sending excerpts to external models.

## Tests

```bash
python -m pytest -q
python -m pytest -q --cov=researchgap --cov-report=term-missing
python -m examples.demo_phase5_offline
python -m examples.demo_phase6_offline
python -m examples.demo_phase7_offline
```

`demo_phase7_offline` builds the Phase 7 Markdown, print-ready HTML and JSON report using fictional papers and simulated responses, without installing LangGraph or connecting any external API. `demo_phase5_offline` runs the pre-existing five-stage simulation without LangGraph. `demo_phase6_offline` constructs and invokes the **actual** LangGraph graph but uses deterministic fictional LLM and literature responses; it requires that LangGraph be installed. Both demos use only fictional sources. Neither assesses actual scientific novelty or external API reliability.

The Phase 6 tests exercise all graph routing/agent contracts through a test-only graph execution double; the separate `test_live_langgraph_contract.py` checks real compiled LangGraph + `InMemorySaver` when the package is installed. In this build environment LangGraph, Streamlit, and live external API access were unavailable; the real-LangGraph contract test therefore **skipped**. See `TEST_REPORT.md` for exact results and tests that still require a connected environment.

## Project structure

```text
ResearchGap-AI/
├── app.py
├── researchgap/
│   ├── agents/                # Existing four source-grounded agents
│   ├── core/
│   │   └── workflow_schemas.py # Phase 6 LangGraph shared-state/result contracts
│   ├── integrations/          # Groq and OpenAlex HTTP clients
│   ├── services/
│   │   └── workflow_service.py # Phase 6 graph, routing, revision and checkpoints
│   ├── tools/                 # Page-linked PDF extraction/chunking
│   ├── ui/dashboard.py        # Phase 7 responsive dark-theme workspace
│   ├── reports/generator.py   # Phase 7 source-grounded Markdown/HTML/JSON
│   ├── report_cli.py          # Phase 7 offline report export
│   └── workflow_cli.py        # Phase 6 one-command CLI
├── examples/
│   ├── sample_papers/         # Fictional source PDFs
│   ├── demo_phase5_offline.py
│   ├── demo_phase6_offline.py
│   ├── demo_phase7_offline.py
│   └── sample_reports/       # Fictional Phase 7 report previews
├── tests/
│   ├── langgraph_test_double.py # TEST-ONLY, NEVER production substitute
│   ├── test_workflow_service.py
│   ├── test_workflow_cli.py
│   ├── test_phase6_ui.py
│   ├── test_phase7_ui.py
│   ├── test_phase7_reports.py
│   └── test_live_langgraph_contract.py
├── .streamlit/config.toml      # Phase 7 dark palette and sizing
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── TEST_REPORT.md
```

## Current limitations / roadmap

- PDF extraction requires selectable text; OCR is outside the MVP. Long papers use explicit context limits.
- External scholarly search only checks a bounded selection of indexed works, not the entire literature. An abstract that supports a model verdict can still be misinterpreted; human paper review is required.
- Verification may be inconclusive if OpenAlex is down or Groq returns unusable data. Results may be partial.
- In-memory checkpoints do not survive application restarts, nor do they provide scalable job management.
- Phase 7: **implemented** a redesigned Streamlit workspace and three-format report exports. Live visual browser verification remains pending.
- Phase 8: evaluation on real research papers, quality benchmarking against single-LLM baseline, security testing, deployment and CI.

### Relevant documentation

- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [OpenAlex API](https://help.openalex.org/api/)
- [Groq API](https://console.groq.com/docs/overview)

## Phase 8 status

Available offline business tests and mocked UI tests pass. Running actual Streamlit, LangGraph, Groq, OpenAlex and human-labeled scientific experiments is **not verified** in the offline environment; consult `TEST_REPORT.md` and `evaluation/RELEASE_CHECKLIST.md`.
