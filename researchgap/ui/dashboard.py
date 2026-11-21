"""Presentation-only layer; never invokes models, external APIs, or file uploads."""

from html import escape

from researchgap.reports import generate_research_report


DASHBOARD_CSS = r"""
<style>
:root { color-scheme: dark; }
.stApp {
  background:
    radial-gradient(ellipse 880px 510px at 95% -180px,rgba(104,80,216,.19),transparent 75%),
    radial-gradient(ellipse 520px 380px at -90px 380px,rgba(41,155,199,.06),transparent 78%),
    #0c1221;
  color: #e4eaf7;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1360px; padding-top: 2.15rem; padding-bottom: 5rem; }
section[data-testid="stSidebar"] {
 background: #0f172a;
 border-right: 1px solid rgba(160,174,206,.10);
}
section[data-testid="stSidebar"] .block-container {padding-top: 2rem;}
[data-testid="stMetric"] {
 background:linear-gradient(140deg,#18223a 0%,#151e33 95%);
 border:1px solid #2b3550; border-radius:16px; padding:17px 19px;
 min-height:105px;
}
[data-testid="stMetricLabel"]{color:#99aac9;font-weight:600;font-size:.86rem}
[data-testid="stMetricValue"]{color:#f6f7fc;font-weight:750;letter-spacing:-.035em}
[data-testid="stFileUploader"]{border:1px dashed #665eab;border-radius:16px;padding:12px;background:#131c31}
[data-testid="stFileUploaderDropzone"] {border:0;background:transparent;border-radius:12px;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
 gap:.3rem;overflow-x:auto;border-bottom:1px solid #303b59;padding:4px 3px 12px;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
 border:1px solid transparent;border-radius:10px;color:#a7b5d3;
 padding:10px 15px;font-weight:650;font-size:.90rem;white-space:nowrap;
}
[data-testid="stTabs"] button[aria-selected="true"] {
 color:#ede7ff;background:#352956;border-color:#7160ac;
}
[data-testid="stTabs"] [data-baseweb="tab-panel"]{padding-top:23px}
[data-testid="stExpander"]{border:1px solid #2f3a56;border-radius:12px;background:#141d31;overflow:hidden}
[data-testid="stExpander"] summary{font-weight:700}
[data-testid="stAlert"]{border-radius:12px;border:1px solid #3a4964}
[data-testid="stDownloadButton"] button{border-radius:11px;border-color:#495678;color:#e9ebfd;font-weight:650}
.stButton button[kind="primary"] {
 background:linear-gradient(110deg,#8260fa,#584dd9) !important;color:#fff !important;
 box-shadow:0 8px 25px #4f46c43c;border:1px solid #9280fa !important;
 border-radius:11px;font-weight:750;min-height:46px;
}
.stButton button[kind="secondary"]{border-radius:10px;border:1px solid #455073;color:#dee5ff;font-weight:650;min-height:40px}
.stButton button:focus-visible,.stDownloadButton button:focus-visible {outline:3px solid #d1c5ff!important;outline-offset:2px}
[data-testid="stDataFrame"], table{border-radius:12px;overflow:hidden}
h1,h2,h3 {letter-spacing:-.026em}
h2 {font-size:1.45rem}
h3{font-size:1.1rem}
p,li{line-height:1.7}
[data-testid="stCaptionContainer"]{color:#92a4c4}
.rg-topline{display:flex;align-items:center;gap:10px;margin-bottom:14px;color:#a2b2d6;font-size:12px;letter-spacing:2.2px;text-transform:uppercase;font-weight:800}
.rg-mark{display:inline-flex;align-items:center;justify-content:center;width:27px;height:27px;border-radius:9px;background:#6d5bdb;color:white;font-size:17px;letter-spacing:0}
.rg-hero{position:relative;overflow:hidden;border:1px solid #384268;border-radius:22px;padding:39px 42px 34px;margin-bottom:24px;background:linear-gradient(120deg,#19233c 0%,#1c2440 49%,#2c2354 100%);}
.rg-hero:before{content:'';position:absolute;right:-110px;top:-175px;width:410px;height:410px;border:1px solid #ae9cff70;border-radius:50%;box-shadow:0 0 0 41px #ae9cff0e,0 0 0 92px #b6a3fa0b}
.rg-hero:after{content:'';position:absolute;right:105px;bottom:-90px;width:170px;height:170px;background:radial-gradient(circle,#8d74ff33,transparent 70%);border-radius:50%}
.rg-kicker{font-size:11px;font-weight:800;letter-spacing:2.7px;color:#b9acff;text-transform:uppercase;margin-bottom:11px}
.rg-hero h1{font-size:clamp(2rem,3.5vw,3.15rem);font-weight:830;letter-spacing:-1.7px;line-height:1.12;color:#f5f4ff;margin:0 0 13px;max-width:820px}
.rg-hero p{max-width:710px;color:#bac9e7;font-size:15px;margin:0;line-height:1.65}
.rg-pills{display:flex;gap:9px;flex-wrap:wrap;margin-top:23px}
.rg-pill{background:#ffffff0c;border:1px solid #ffffff27;border-radius:999px;padding:6px 12px;font-size:12px;font-weight:650;color:#dad8fa}
.rg-section-label{display:flex;align-items:center;gap:11px;color:#dce3f6;font-size:14px;font-weight:750;margin:7px 0 14px}
.rg-section-dot{width:8px;height:8px;border-radius:50%;background:#927cff;box-shadow:0 0 0 4px #927cff20}
.rg-panel{background:linear-gradient(140deg,#17213a,#141c31);border:1px solid #303b58;border-radius:16px;padding:22px 24px;margin:16px 0 18px}
.rg-panel-title{font-size:15px;font-weight:780;color:#e9e8ff;margin-bottom:5px}
.rg-panel-desc{font-size:13px;color:#a6b6d3;line-height:1.65}
.rg-steps{display:grid;grid-template-columns:repeat(6,1fr);gap:9px;margin:19px 0 25px}
.rg-step{background:#111a2e;border:1px solid #2d3853;border-radius:13px;padding:13px 10px;min-height:88px}
.rg-step.active{border-color:#7c67e5;background:#241f49;box-shadow:inset 0 1px 0 #ffffff18}
.rg-step.done{border-color:#3c6a76;background:#152c35}
.rg-step .num{color:#9caeca;font-size:10px;font-weight:800;letter-spacing:1px;margin-bottom:8px}
.rg-step .text{color:#dde6fa;font-weight:720;font-size:12px;line-height:1.3}
.rg-step.done .num{color:#72d9c5}
.rg-footnote{padding:15px 18px;background:#171e34;border:1px solid #343c58;border-radius:12px;color:#b8c4e2;font-size:12px;margin-top:22px}
.rg-side-brand{font-weight:850;font-size:20px;letter-spacing:-.6px;color:#f5f2ff;margin:7px 0 4px}
.rg-sidebar-hr{height:1px;background:#303a55;margin:20px 0}
.rg-side-kicker{font-size:10px;letter-spacing:2px;color:#9c8df9;font-weight:850;text-transform:uppercase}
.rg-side-line{padding:9px 10px;font-size:13px;color:#b5c3df;border-radius:9px;border:1px solid transparent}
.rg-side-line strong{color:#ecebff}
.rg-preview-label{font-size:12px;letter-spacing:1.4px;color:#b3a7ff;font-weight:800;text-transform:uppercase;margin:8px 0}
@media(max-width:1000px){.rg-steps{grid-template-columns:repeat(3,1fr)}.rg-hero{padding:30px 28px}}
@media(max-width:620px){.block-container{padding:1rem 1rem 4rem}.rg-hero{padding:26px 22px}.rg-steps{grid-template-columns:repeat(2,1fr)}.rg-hero h1{letter-spacing:-.9px}.rg-hero:before{opacity:.3}}
</style>
"""


def render_header(st) -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="rg-topline"><span class="rg-mark">✦</span> RESEARCHGAP / INTELLIGENCE WORKSPACE</div>
    <div class="rg-hero">
       <div class="rg-kicker">Research intelligence · Powered by 4 specialized agents</div>
       <h1>From research papers<br>to research possibilities.</h1>
       <p>Analyze, compare, and challenge candidate research gaps — with traceable
       source quotations and transparent scientific uncertainty.</p>
       <div class="rg-pills"><span class="rg-pill">01 / Evidence-grounded</span>
       <span class="rg-pill">02 / Multi-agent reasoning</span>
       <span class="rg-pill">03 / Human-reviewed novelty</span></div>
    </div>""", unsafe_allow_html=True)


def render_sidebar(st, has_key: bool) -> None:
    with st.sidebar:
        st.markdown('<div class="rg-side-kicker">WORKSPACE / V1.0</div><div class="rg-side-brand">✦ ResearchGap AI</div>', unsafe_allow_html=True)
        st.caption("Academic literature intelligence")
        st.markdown('<div class="rg-sidebar-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="rg-side-kicker">PIPELINE</div>', unsafe_allow_html=True)
        for stage in ("01  Import papers", "02  Analyze evidence", "03  Compare studies", "04  Discover gaps", "05  Verify literature", "06  Orchestrate agents", "07  Export report", "08  Evaluate evidence"):
            st.markdown(f'<div class="rg-side-line">{escape(stage)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="rg-sidebar-hr"></div>', unsafe_allow_html=True)
        st.markdown('<div class="rg-side-kicker">MODEL CONNECTION</div>', unsafe_allow_html=True)
        from researchgap.core.analysis_config import get_llm_configuration_summary
        cfg_summary = get_llm_configuration_summary()
        if cfg_summary["has_keys"] or has_key:
            st.success(f"Connected: {cfg_summary['summary']}")
        else:
            st.info("Add GEMINI_API_KEYS or GROQ_API_KEY to .env to enable agents")
        st.caption("OpenAlex provides literature metadata. Source documents and queries are sent to external APIs only after you request an AI action.")


def render_progress(st) -> None:
    state = st.session_state
    statuses = [
        ("Import", bool(state.get("processing_result"))),
        ("Analysis", bool(state.get("analysis_result"))),
        ("Comparison", bool(state.get("comparison_result"))),
        ("Discovery", bool(state.get("gap_result"))),
        ("Verification", bool(state.get("verification_result"))),
        ("Workflow", bool(state.get("workflow_result"))),
    ]
    cards = []
    for i, (label, complete) in enumerate(statuses, 1):
        css = "done" if complete else ("active" if i == 1 or statuses[i - 2][1] else "")
        marker = "✓ FINISHED" if complete else f"0{i} / STAGE"
        cards.append(f'<div class="rg-step {css}"><div class="num">{marker}</div><div class="text">{escape(label)}</div></div>')
    st.markdown('<div class="rg-section-label"><span class="rg-section-dot"></span> Your analysis pipeline</div>'
                + '<div class="rg-steps">' + ''.join(cards) + '</div>', unsafe_allow_html=True)


def render_report_studio(st) -> None:
    state = st.session_state
    batch = state.get("processing_result")
    manual = {"analyses": state.get("analysis_result"),
              "comparison": state.get("comparison_result"),
              "gaps": state.get("gap_result"),
              "verification": state.get("verification_result")}
    workflow = state.get("workflow_result")
    if not any((batch, workflow, *manual.values())):
        st.info("Upload and process research PDFs first. Your export studio will become available here.")
        return

    st.markdown('<div class="rg-panel"><div class="rg-panel-title">Your research deliverable</div>'
                '<div class="rg-panel-desc">Generate a polished, source-traceable report in Markdown, '
                'print-ready HTML, and a complete machine-readable JSON package. '
                'The report is assembled from existing results; no additional model calls are made.</div></div>', unsafe_allow_html=True)
    source_objects = (batch, workflow, *manual.values())
    signature = tuple(id(x) if x is not None else None for x in source_objects)
    if state.get("report_source_signature") != signature:
        state.pop("report_artifacts", None)
        state["report_source_signature"] = signature

    st.caption("Select the report identity; do not include private information in reports you plan to publish.")
    report_title = st.text_input("Report title", value="ResearchGap AI — Literature Review", max_chars=120, key="report_title")
    report_author = st.text_input("Prepared by (optional)", value="", max_chars=100, key="report_author")
    st.caption("When a full workflow result is available, its source-consistent stages take priority over manual results.")
    if st.button("Generate research report", type="primary", key="generate_research_report"):
        try:
            state["report_artifacts"] = generate_research_report(
                batch=batch, workflow=workflow,
                **({} if workflow is not None else manual),
                title=report_title, author=report_author,
            )
            state["report_identity"] = (report_title, report_author)
        except ValueError as exc:
            st.error(f"Cannot generate a research report: {exc}")
    output = state.get("report_artifacts")
    if output is None:
        st.info("Choose Generate research report to prepare all three formats.")
        return
    if state.get("report_identity") != (report_title, report_author):
        st.info("Report details changed. Generate the report again to update the exports.")
        return
    st.success("Report generated from saved research outputs — no API requests were made.")
    a, b, c = st.columns(3)
    with a:
        st.download_button("⬇ Download Markdown", data=output.markdown.encode("utf-8"),
                           file_name=output.filename_stem + ".md", mime="text/markdown",
                           use_container_width=True)
    with b:
        st.download_button("⬇ Download HTML", data=output.html.encode("utf-8"),
                           file_name=output.filename_stem + ".html", mime="text/html",
                           use_container_width=True)
    with c:
        st.download_button("⬇ Download JSON", data=output.json_text.encode("utf-8"),
                           file_name=output.filename_stem + ".json", mime="application/json",
                           use_container_width=True)
    st.markdown('<div class="rg-preview-label">Report preview</div>', unsafe_allow_html=True)
    with st.expander("Preview report contents", expanded=True):
        preview = output.markdown[:12000]
        st.markdown(preview)
        if len(output.markdown) > len(preview):
            st.caption("Preview shortened to 12,000 characters. Downloads contain the entire report.")
    st.info("To make a PDF, open the downloaded HTML file in a browser and choose Print → Save as PDF.")


def render_evaluation_studio(st) -> None:
    """Phase 8 controlled single-pass baseline with explicit user consent.

    This is a downstream comparison using the SAME Phase 2 facts as the
    multi-agent workflow, not a full scientific benchmark. Human-reviewed
    correctness remains unmeasured until annotations are supplied.
    """
    from researchgap.evaluation.baseline import SinglePassBaseline
    from researchgap.evaluation.metrics import compare_systems, create_review_template
    from researchgap.evaluation.telemetry import MeteredLLM
    from researchgap.core.analysis_config import configured_api_key
    from researchgap.integrations.groq_client import GroqClient
    import os

    st.markdown(
        '<div class="rg-panel"><div class="rg-panel-title">Evaluation Lab</div>'
        '<div class="rg-panel-desc">Compare the four-agent workflow against one '
        'single-pass gap-generation request using the same extracted Phase 2 source facts. '
        'Automatic checks validate citation integrity; scientific accuracy needs human labels.</div></div>', unsafe_allow_html=True,
    )
    state = st.session_state
    batch, workflow = state.get("processing_result"), state.get("workflow_result")
    signature = (id(batch), id(workflow))
    if state.get("evaluation_source_signature") != signature:
        state.pop("baseline_result", None)
        state.pop("evaluation_result", None)
        state.pop("baseline_telemetry", None)
        state["evaluation_source_signature"] = signature
    if workflow is None or workflow.paper_analyses is None or batch is None:
        st.info("Process papers and run the full agent workflow first. Evaluation uses exactly those source facts.")
        return
    key = configured_api_key()
    if not key:
        st.info("Configure GEMINI_API_KEYS or GROQ_API_KEY to run the single-pass comparison baseline.")
    if st.button("Run single-pass evaluation baseline", key="run_phase8_baseline", disabled=not bool(key)):
        try:
            from researchgap.integrations.failover_client import create_llm_client
            from researchgap.integrations.groq_client import RealGroqClient
            if GroqClient is not RealGroqClient:
                llm = GroqClient(key, model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"))
            else:
                llm = create_llm_client()
            client = MeteredLLM(llm)
            with st.spinner("Running the single-pass baseline using shared source facts..."):
                baseline = SinglePassBaseline(client).run(workflow.paper_analyses)
            state["baseline_result"] = baseline
            state["evaluation_result"] = compare_systems(batch, baseline, workflow)
            state["baseline_telemetry"] = client.measurements()
        except (ValueError, TypeError) as exc:
            st.error(f"Baseline evaluation could not be completed ({type(exc).__name__}).")

    result = state.get("evaluation_result")
    baseline = state.get("baseline_result")
    if not result or not baseline:
        st.caption("No baseline has been run. The primary analysis workflow remains available without evaluation.")
        return
    st.subheader("Citation checks — baseline vs. agents")
    a, b, c = st.columns(3)
    a.metric("Single-pass gaps", result.baseline.gap_count)
    b.metric("Multi-agent gaps", result.multi_agent.gap_count)
    c.metric("Invalid baseline citations", result.baseline.gaps_with_invalid_reference)

    def rate(value):
        return "Not measured" if value is None else f"{value:.1%}"

    traces = state.get("workflow_telemetry")
    baseline_usage = state.get("baseline_telemetry")
    if traces and baseline_usage:
        st.subheader("Measured execution / API calls")
        st.table([
            {"Measurement": "LLM calls", "Single-pass": baseline_usage.get("calls", 0),
             "Multi-agent": traces.get("llm", {}).get("calls", 0)},
            {"Measurement": "LLM request latency (seconds)",
             "Single-pass": baseline_usage.get("llm_latency_seconds", 0),
             "Multi-agent": traces.get("llm", {}).get("llm_latency_seconds", 0)},
            {"Measurement": "OpenAlex searches", "Single-pass": 0,
             "Multi-agent": traces.get("literature_search", {}).get("search_calls", 0)},
        ])
        st.caption("Measured calls and elapsed time, not a price estimate. API cost is unknown without observed token counts and model pricing.")
        import json
        st.download_button("Download execution telemetry (JSON)",
            data=json.dumps({"baseline": baseline_usage, "multi_agent": traces}, indent=2).encode("utf-8"),
            file_name="researchgap_api_telemetry.json", mime="application/json")
    st.table([
        {"Metric": "Exact source-page citation rate",
         "Single-pass": rate(result.baseline.page_citation_integrity_rate),
         "Multi-agent": rate(result.multi_agent.page_citation_integrity_rate)},
        {"Metric": "Gaps citing two papers",
         "Single-pass": rate(result.baseline.cross_paper_citation_rate),
         "Multi-agent": rate(result.multi_agent.cross_paper_citation_rate)},
        {"Metric": "Human-reviewed evidence support",
         "Single-pass": rate(result.baseline.human_support_rate),
         "Multi-agent": rate(result.multi_agent.human_support_rate)},
    ])
    if baseline.status != "complete":
        st.warning("Baseline could not run completely; results are not a valid success comparison.")
    st.warning("A correct quote is not proof of a valid scientific claim. No human accuracy or novelty result has been inferred.")
    st.download_button("Download evaluation metrics (JSON)",
        data=result.model_dump_json(indent=2).encode("utf-8"),
        file_name="researchgap_evaluation_metrics.json", mime="application/json")
    st.download_button("Download human review template (CSV)",
        data=create_review_template(baseline, workflow).encode("utf-8"),
        file_name="researchgap_human_review.csv", mime="text/csv")
    st.download_button("Download baseline outputs (JSON)",
        data=baseline.model_dump_json(indent=2).encode("utf-8"),
        file_name="researchgap_single_pass_baseline.json", mime="application/json")
    st.caption("For human-reviewed scores, annotate the CSV with yes/no evidence-support labels, "
               "then run `python -m researchgap.evaluation.cli` using saved Phase 1, baseline and workflow JSON.")
