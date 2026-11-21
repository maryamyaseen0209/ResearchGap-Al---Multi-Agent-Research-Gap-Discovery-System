"""ResearchGap AI — Phase 7 research workspace atop the original 6-phase backend."""

import streamlit as st

from researchgap.core.config import MAX_PAPERS, MAX_PDF_SIZE_MB, MIN_PAPERS
from researchgap.services.batch_processor import PaperUpload, process_batch
from researchgap.core.analysis_config import configured_api_key, get_llm_configuration_summary
from researchgap.agents.paper_analyzer import CATEGORY_TO_ATTRIBUTE, PaperAnalyzer
from researchgap.integrations.groq_client import GroqClient, RealGroqClient
from researchgap.integrations.failover_client import create_llm_client
from researchgap.services.analysis_service import analyze_batch
from researchgap.agents.paper_comparator import PaperComparator
from researchgap.services.comparison_service import compare_batch
from researchgap.agents.gap_discoverer import GapDiscoverer
from researchgap.services.gap_service import discover_gaps
from researchgap.agents.gap_verifier import GapVerifier
from researchgap.integrations.openalex_client import OpenAlexClient
from researchgap.services.verification_service import verify_gaps
from researchgap.services.workflow_service import ResearchWorkflow, WorkflowDependencyError
from researchgap.ui.dashboard import render_header, render_progress, render_report_studio, render_sidebar, render_evaluation_studio
import os
from time import perf_counter
from researchgap.evaluation.telemetry import MeteredLLM, MeteredSearch

st.set_page_config(page_title="ResearchGap AI | Evidence-Grounded Literature Review", page_icon="📚", layout="wide")


def get_app_llm(model_override: str | None = None):
    """Returns an LLM client: respects test monkeypatches of GroqClient, otherwise creates multi-key failover client."""
    from researchgap.integrations import groq_client
    active_groq_class = getattr(groq_client, "GroqClient", GroqClient)
    if active_groq_class is not RealGroqClient:
        return active_groq_class(
            api_key=configured_api_key() or "test-key",
            model=model_override or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        )
    return create_llm_client()



def clear_results() -> None:
    """Avoid presenting stale results after the upload selection changes."""
    st.session_state.pop("processing_result", None)
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("comparison_result", None)
    st.session_state.pop("gap_result", None)
    st.session_state.pop("verification_result", None)
    st.session_state.pop("workflow_result", None)
    st.session_state.pop("report_artifacts", None)
    st.session_state.pop("baseline_result", None)
    st.session_state.pop("evaluation_result", None)
    st.session_state.pop("baseline_telemetry", None)
    st.session_state.pop("workflow_telemetry", None)


render_header(st)
render_sidebar(st, bool(configured_api_key()))
render_progress(st)
workspace_tabs = st.tabs([
    "01 · Library", "02 · Analyze", "03 · Compare", "04 · Discover",
    "05 · Verify", "06 · Agent workflow", "07 · Reports", "08 · Evaluation",
])

with workspace_tabs[0]:
    st.markdown('<div class="rg-section-label"><span class="rg-section-dot"></span> Add your source literature</div>', unsafe_allow_html=True)
    st.markdown('<div class="rg-panel"><div class="rg-panel-title">A reliable review starts with reliable sources.</div><div class="rg-panel-desc">Add 3–5 original research PDFs. Text and page references remain linked throughout the workflow. Documents are processed locally until you explicitly run an AI action.</div></div>', unsafe_allow_html=True)

    uploads = st.file_uploader(
        "Upload research papers (PDF only)",
        type=["pdf"],
        accept_multiple_files=True,
        on_change=clear_results,
        help=f"Upload {MIN_PAPERS} to {MAX_PAPERS} unique PDFs, up to {MAX_PDF_SIZE_MB} MB each.",
    )

    count = len(uploads)
    if count and count < MIN_PAPERS:
        st.info(f"Select at least {MIN_PAPERS} PDFs to process (currently {count}).")
    elif count > MAX_PAPERS:
        st.error(f"Choose no more than {MAX_PAPERS} PDFs (currently {count}).")

    can_process = MIN_PAPERS <= count <= MAX_PAPERS
    if st.button("Process research papers", type="primary", disabled=not can_process):
        with st.spinner("Extracting and chunking papers..."):
            inputs = [PaperUpload(filename=f.name, content=f.getvalue()) for f in uploads]
            st.session_state.processing_result = process_batch(inputs)
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("comparison_result", None)
            st.session_state.pop("gap_result", None)
            st.session_state.pop("verification_result", None)
            st.session_state.pop("workflow_result", None)

    batch = st.session_state.get("processing_result")
    if batch is not None:
        st.divider()
        st.subheader("Processing summary")
        left, center, right = st.columns(3)
        left.metric("Successfully processed", batch.successful_count)
        center.metric("Text chunks", batch.total_chunks)
        right.metric("Failed uploads", len(batch.failures))

        if batch.failures:
            st.warning("Some PDFs could not be processed:")
            for failure in batch.failures:
                st.error(f"**{failure.filename}** — {failure.message} ({failure.code})")

        if not batch.ready_for_analysis:
            st.warning("Fewer than three unique, readable PDFs remain. Add valid papers for later phases.")
        elif batch.papers:
            st.success("At least three papers are ready for AI paper analysis.")

        if batch.papers:
            st.download_button(
                "Download structured results (JSON)",
                data=batch.model_dump_json(indent=2).encode("utf-8"),
                file_name="researchgap_extracted_papers.json",
                mime="application/json",
            )

        for paper in batch.papers:
            with st.expander(f"📄 {paper.filename} — {paper.page_count} pages, {len(paper.chunks)} chunks"):
                st.write(f"**Document ID:** `{paper.document_id}`")
                st.write(f"**Extracted characters:** {paper.total_characters:,}")
                st.write(f"**Pages with text:** {paper.text_page_count}/{paper.page_count}")
                if paper.title:
                    st.write(f"**PDF title:** {paper.title}")
                for warning in paper.warnings:
                    st.warning(warning)

                page_number = st.selectbox(
                    "Inspect extracted page",
                    options=[page.page_number for page in paper.pages],
                    key=f"page_{paper.document_id}",
                )
                page = paper.pages[page_number - 1]
                st.text_area(
                    f"Extracted text — page {page_number}",
                    value=page.text if page.text else "(No selectable text on this page)",
                    height=200,
                    disabled=True,
                    key=f"text_{paper.document_id}_{page_number}",
                )
                if paper.chunks:
                    chunk_index = st.selectbox(
                        "Inspect evidence-linked chunk",
                        options=range(len(paper.chunks)),
                        format_func=lambda i: (
                            f"Chunk {paper.chunks[i].chunk_index} · "
                            f"page {paper.chunks[i].page_number}"
                        ),
                        key=f"chunk_{paper.document_id}",
                    )
                    chunk = paper.chunks[chunk_index]
                    st.caption(
                        f"Chunk ID: {chunk.chunk_id} | Page: {chunk.page_number} | "
                        f"Text offsets: [{chunk.start_char}:{chunk.end_char}]"
                    )
                    st.code(chunk.text, language="text")



with workspace_tabs[1]:
    if st.session_state.get("processing_result") is None:
        st.info("Start by processing 3–5 papers in the Library tab.")

    if batch is not None and batch.papers:
        st.divider()
        st.subheader("🤖 Phase 2 — Paper Analyzer Agent")
        st.caption(
            "The AI extracts objectives, methods, datasets, findings, contributions, and limitations. "
            "Each accepted fact must have an exact quote matched to the original extracted PDF. "
            "PDF excerpts are sent to Groq only when you click Analyze."
        )
        api_key = configured_api_key()
        if not api_key:
            st.info("To enable Phase 2, set GEMINI_API_KEYS or GROQ_API_KEYS in .env; then restart the app.")
        else:
            try:
                active_llm = get_app_llm()
                backend_info = active_llm.status_summary() if hasattr(active_llm, "status_summary") else getattr(active_llm, "model", "default")
                st.caption(f"⚡ **Active LLM Provider:** {backend_info}")
            except Exception:
                pass

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                run_analyze = st.button("Analyze uploaded papers with AI", disabled=not batch.ready_for_analysis)
            with btn_col2:
                analyzed_state = st.session_state.get("analysis_result")
                is_partial = False
                if analyzed_state and analyzed_state.analyses:
                    is_partial = any(
                        a.status in ("partial", "failed")
                        or a.coverage.batches_succeeded < a.coverage.batches_attempted
                        for a in analyzed_state.analyses
                    )
                run_resume = st.button("Resume incomplete analysis", disabled=not (batch.ready_for_analysis and is_partial))

            if run_analyze or run_resume:
                with st.spinner("Analyzing papers with evidence-grounded AI extraction..."):
                    llm_client = get_app_llm()
                    if run_resume and hasattr(llm_client, "reset_transient_limits"):
                        llm_client.reset_transient_limits()
                    analyzer = PaperAnalyzer(llm_client, enable_cache=True)
                    st.session_state.analysis_result = analyze_batch(batch, analyzer)
                    st.session_state.pop("comparison_result", None)
                    st.session_state.pop("gap_result", None)
                    st.session_state.pop("verification_result", None)
                    st.session_state.pop("workflow_result", None)

        analyzed = st.session_state.get("analysis_result")
        if analyzed is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Completed Papers", analyzed.completed_count)
            m2.metric("Partial Papers", analyzed.partial_count)
            m3.metric("Failed Papers", analyzed.failed_count)
            m4.metric("Accepted Facts", analyzed.total_accepted_facts)
            st.caption(
                f"📊 **Global Batch Progress:** {analyzed.total_batches_succeeded}/{analyzed.total_batches_planned} "
                f"batches completed across papers."
            )
            for failure in analyzed.failures:
                st.error(f"{failure.filename}: {failure.code} — {failure.message}")
            for item in analyzed.analyses:
                status_icon = (
                    "✅ Complete" if item.is_complete
                    else ("⚠️ Partial (Resumable)" if item.is_partial else "❌ Failed (0 Batches Succeeded)")
                )
                with st.expander(f"🧪 {item.filename} — {status_icon} ({item.verified_fact_count} cited facts, {item.supported_fact_count} fully supported)"):
                    st.write(f"**Title:** {item.title}")
                    st.write(f"**Model Provenance:** {item.model}")

                    # Detailed coverage and batch progress breakdown
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Selected Coverage", f"{item.coverage.selected_coverage_pct}%")
                    c2.metric("Document Coverage", f"{item.coverage.document_coverage_pct}%")
                    c3.metric("Batches Succeeded", f"{item.coverage.batches_succeeded}/{item.coverage.batches_attempted}")
                    c4.metric("Cached Batches", item.coverage.batches_cached)

                    st.caption(
                        f"**Text Coverage:** {item.coverage.unique_analyzed_characters:,} / {item.coverage.total_characters:,} "
                        f"unique characters across {item.coverage.analyzed_chunks}/{item.coverage.total_chunks} chunks."
                    )
                    for warning in item.warnings:
                        st.warning(warning)
                    labels = {
                        "research_objectives": "Research objectives", "methodologies": "Methodologies",
                        "datasets": "Datasets", "findings": "Findings",
                        "limitations": "Limitations", "contributions": "Contributions",
                    }
                    for attr in CATEGORY_TO_ATTRIBUTE.values():
                        facts = getattr(item, attr)
                        st.write(f"**{labels[attr]}**")
                        if not facts:
                            st.caption("No source-verified facts extracted in analyzed chunks.")
                        for fact in facts:
                            support_badge = (
                                "🟢 [Supported]" if fact.support_status == "supported"
                                else ("🟡 [Partially Supported]" if fact.support_status == "partially_supported"
                                      else "🟠 [Needs Review]")
                            )
                            st.markdown(f"• {fact.statement} `{support_badge}`")
                            ev = fact.evidence
                            st.caption(
                                f"Evidence: {ev.filename}, page {ev.page_number}, "
                                f"offsets [{ev.start_char}:{ev.end_char}] — “{ev.quote}”"
                            )
                            if fact.additional_evidence:
                                for aev in fact.additional_evidence:
                                    st.caption(
                                        f"Additional Evidence: {aev.filename}, page {aev.page_number}, "
                                        f"offsets [{aev.start_char}:{aev.end_char}] — “{aev.quote}”"
                                    )
            st.download_button(
                "Download AI paper analyses (JSON)",
                data=analyzed.model_dump_json(indent=2).encode("utf-8"),
                file_name="researchgap_paper_analyses.json",
                mime="application/json",
            )




with workspace_tabs[2]:
    if st.session_state.get("processing_result") is None:
        st.info("Run the Paper Analyzer first to unlock comparison.")

    if batch is not None and batch.papers:
        analyzed = st.session_state.get("analysis_result")
        if analyzed is not None:
            st.divider()
            st.subheader("🔎 Phase 3 — Paper Comparison Agent")
            st.caption(
                "Compare methodologies, datasets, findings, limitations, and objectives across "
                "3–5 analyzed papers. Comparison claims cite source-verified Phase 2 facts; "
                "the source fact must still be checked for scientific meaning."
            )
            api_key = configured_api_key()
            eligible = 3 <= analyzed.successful_count <= 5
            if not eligible:
                st.warning("Paper comparison requires at least three successful, distinct paper analyses.")
            if not api_key:
                st.info("Configure GROQ_API_KEY in .env to enable the Phase 3 comparison agent.")
            elif st.button("Compare analyzed papers", disabled=not eligible):
                with st.spinner("Comparing papers with evidence-linked AI reasoning..."):
                    comparator = PaperComparator(get_app_llm())
                    try:
                        st.session_state.comparison_result = compare_batch(analyzed, comparator)
                        st.session_state.pop("workflow_result", None)
                        st.session_state.pop("gap_result", None)
                        st.session_state.pop("verification_result", None)
                    except ValueError as exc:
                        st.error(f"Cannot compare these analyses: {exc}")
                        st.session_state.pop("workflow_result", None)
                        st.session_state.pop("comparison_result", None)
                        st.session_state.pop("gap_result", None)
                        st.session_state.pop("verification_result", None)

            comparison = st.session_state.get("comparison_result")
            if comparison is not None:
                left, center, right = st.columns(3)
                left.metric("Compared papers", comparison.paper_count)
                center.metric("Grounded cross-paper insights", len(comparison.insights))
                right.metric("Comparison status", comparison.status)
                st.write(
                    f"**Evidence coverage:** {comparison.coverage.facts_sent_to_llm}/"
                    f"{comparison.coverage.total_source_facts} extracted facts included in AI context"
                )
                for warning in comparison.warnings:
                    st.warning(warning)

                st.write("**Paper-by-paper comparison table**")
                categories = (
                    ("objectives", "Objectives"), ("methodologies", "Methodologies"),
                    ("datasets", "Datasets"), ("findings", "Findings"),
                    ("limitations", "Limitations"), ("contributions", "Contributions"),
                )
                st.table([
                    {
                        "Paper": row.title,
                        "Status": row.analysis_status,
                        **{
                            label: "; ".join(f.statement for f in getattr(row, attr)) or "Not extracted"
                            for attr, label in categories
                        },
                    }
                    for row in comparison.comparison_table
                ])
                for row in comparison.comparison_table:
                    with st.expander(f"📑 Sources — {row.filename}"):
                        for attr, label in categories:
                            facts = getattr(row, attr)
                            st.write(f"**{label}:**")
                            if not facts:
                                st.caption("No verified extracted facts (not proof of absence).")
                            for fact in facts:
                                st.write(f"• {fact.statement}")
                                ev = fact.evidence
                                st.caption(
                                    f"{fact.fact_id} · page {ev.page_number} · "
                                    f"offsets [{ev.start_char}:{ev.end_char}] · “{ev.quote}”"
                                )
                st.write("**Cross-paper insights**")
                if not comparison.insights:
                    st.info("No cross-paper comparisons passed citation checks. Review the paper analyses.")
                for insight in comparison.insights:
                    st.write(f"**{insight.kind.replace('_', ' ').title()}** — {insight.statement}")
                    for source in insight.cited_facts:
                        ev = source.evidence
                        st.caption(
                            f"{source.filename} · {source.fact_id} · page {ev.page_number}: “{ev.quote}”"
                        )
                st.download_button(
                    "Download paper comparison (JSON)",
                    data=comparison.model_dump_json(indent=2).encode("utf-8"),
                    file_name="researchgap_paper_comparison.json",
                    mime="application/json",
                )



with workspace_tabs[3]:
    if st.session_state.get("processing_result") is None:
        st.info("Compare analyzed papers before discovering gaps.")

    # Phase 4 remains an explicit, user-triggered LLM action. No papers are sent
    # merely because a user uploads files, and stale gaps are invalidated upstream.
    comparison = st.session_state.get("comparison_result")
    if comparison is not None:
        st.divider()
        st.subheader("💡 Phase 4 — Research Gap Discovery Agent")
        st.caption(
            "Generate 3–5 candidate research directions when sufficient evidence exists. "
            "Every accepted proposal must cite facts from two different uploaded papers, "
            "including at least one explicitly extracted limitation. These are hypotheses, "
            "NOT verified claims of scientific novelty."
        )
        api_key = configured_api_key()
        if not api_key:
            st.info("Configure GROQ_API_KEY in .env to enable the Phase 4 discovery agent.")
        elif st.button("Discover candidate research gaps", disabled=comparison.paper_count < 3):
            with st.spinner("Generating source-linked research questions with the LLM..."):
                try:
                    finder = GapDiscoverer(get_app_llm())
                    st.session_state.gap_result = discover_gaps(comparison, finder)
                    st.session_state.pop("workflow_result", None)
                    st.session_state.pop("workflow_telemetry", None)
                    st.session_state.pop("verification_result", None)
                except ValueError as exc:
                    st.error(f"Cannot discover gaps from this comparison: {exc}")
                    st.session_state.pop("workflow_result", None)
                    st.session_state.pop("gap_result", None)
                    st.session_state.pop("verification_result", None)

        gaps = st.session_state.get("gap_result")
        if gaps is not None:
            left, center, right = st.columns(3)
            left.metric("Candidate gaps", len(gaps.gaps))
            center.metric("Papers considered", gaps.paper_count)
            right.metric("Discovery status", gaps.status)
            if st.session_state.get("verification_result") is None:
                st.warning(
                    "External literature verification has NOT been performed. A missing topic "
                    "in these uploaded papers does not prove that it is unexplored."
                )
            st.write(
                f"**Evidence in discovery context:** {gaps.coverage.facts_sent_to_llm}/"
                f"{gaps.coverage.total_source_facts} source-linked facts"
            )
            for warning in gaps.warnings:
                st.warning(warning)
            if not gaps.gaps:
                st.info("No gaps passed evidence validation. Try more complete, relevant papers.")
            for gap in gaps.gaps:
                with st.expander(f"🔬 {gap.gap_id}: {gap.title}"):
                    st.write(f"**Type:** {gap.kind.replace('_', ' ').title()}")
                    st.write(f"**Why investigate it:** {gap.rationale}")
                    st.write("**Research questions:**")
                    for q in gap.research_questions:
                        st.write(f"• {q.text}")
                    st.write(f"**Potential importance (heuristic):** {gap.importance} — {gap.importance_reason}")
                    st.write(f"**Preliminary feasibility (heuristic):** {gap.feasibility} — {gap.feasibility_reason}")
                    st.write("**Source evidence:**")
                    for fact in gap.cited_facts:
                        ev = fact.evidence
                        st.write(f"• {fact.statement}")
                        st.caption(
                            f"{fact.filename} · {fact.fact_id} · page {ev.page_number} · "
                            f"offsets [{ev.start_char}:{ev.end_char}] · “{ev.quote}”"
                        )
                    if st.session_state.get("verification_result") is None:
                        st.caption("Literature novelty verification: NOT PERFORMED until you run Phase 5 below.")
            st.download_button(
                "Download candidate research gaps (JSON)",
                data=gaps.model_dump_json(indent=2).encode("utf-8"),
                file_name="researchgap_candidate_gaps.json",
                mime="application/json",
            )



with workspace_tabs[4]:
    if st.session_state.get("processing_result") is None:
        st.info("Discover candidate gaps before verifying them.")

    # Phase 5 is an independent explicit action; OpenAlex is contacted only on click.
    gaps = st.session_state.get("gap_result")
    if gaps is not None:
        st.divider()
        st.subheader("🛡️ Phase 5 — Research Gap Verification Agent")
        st.caption(
            "Search OpenAlex for related scientific studies, then ask the reviewer agent "
            "whether each returned abstract may challenge the proposed gap. "
            "OpenAlex search queries and returned abstracts are shared with Groq for review. "
            "This is a scoped literature check, NOT a proof of scientific novelty."
        )
        api_key = configured_api_key()
        if not api_key:
            st.info("Configure GROQ_API_KEY in .env to enable literature evidence assessment.")
        elif st.button("Verify candidate gaps against OpenAlex", disabled=not bool(gaps.gaps)):
            with st.spinner("Searching OpenAlex and evaluating retrieved scientific abstracts..."):
                verifier = GapVerifier(
                    get_app_llm(),
                    OpenAlexClient(api_key=os.getenv("OPENALEX_API_KEY", "").strip() or None),
                )
                st.session_state.verification_result = verify_gaps(gaps, verifier)
                st.session_state.pop("workflow_result", None)
        verification = st.session_state.get("verification_result")
        if verification is not None:
            left, center, right = st.columns(3)
            left.metric("Gaps evaluated", len(verification.verifications))
            center.metric("Potential challenges", sum(v.status == "challenged" for v in verification.verifications))
            right.metric("Verification status", verification.status)
            st.warning("A successful search never proves a topic is unexplored. Human review is required.")
            for warning in verification.warnings:
                st.warning(warning)
            for item in verification.verifications:
                with st.expander(f"📖 {item.gap_id} — {item.status.replace('_', ' ').title()}"):
                    st.write(f"**Candidate:** {item.title}")
                    st.write("**Searches:** " + "; ".join(item.search_queries))
                    st.write(
                        f"**OpenAlex searches:** {item.searches_succeeded}/{item.searches_attempted} succeeded; "
                        f"{item.works_retrieved} retrieved works"
                    )
                    for warning in item.warnings:
                        st.warning(warning)
                    if not item.works:
                        st.info("No scholarly records retrieved for this gap, or OpenAlex was unavailable.")
                    evidence_by_id = {a.openalex_id: a for a in item.assessments}
                    for work in item.works:
                        st.write(f"**{work.title}** ({work.publication_year or 'year unknown'}) — {work.url}")
                        if work.doi:
                            st.caption(f"DOI: {work.doi}")
                        assessment = evidence_by_id.get(work.openalex_id)
                        if assessment:
                            st.write(f"**Assessment:** {assessment.relevance} — {assessment.reasoning}")
                            if assessment.evidence_quote:
                                st.caption(f"Quoted abstract/title evidence: “{assessment.evidence_quote}”")
                        else:
                            st.caption("Not independently assessed; evidence insufficient.")
            st.download_button(
                "Download research gap verification (JSON)",
                data=verification.model_dump_json(indent=2).encode("utf-8"),
                file_name="researchgap_literature_verification.json",
                mime="application/json",
            )





with workspace_tabs[5]:
    if st.session_state.get("processing_result") is None:
        st.info("First process at least three source PDFs in the Library tab.")

    # Phase 6: optional ONE-CLICK LangGraph flow; earlier individual actions remain.
    # File upload alone never triggers an external request. Users explicitly consent
    # to processing PDF excerpts and derived gap queries using Groq and OpenAlex.
    if batch is not None and batch.papers:
        st.divider()
        st.subheader("🧠 Phase 6 — Complete LangGraph Multi-Agent Workflow")
        st.caption(
            "One click runs Paper Analyzer → Comparison → Gap Discovery → OpenAlex "
            "Verification, with up to one evidence-grounded revision if a gap is challenged. "
            "Selected paper text is shared with Groq and scholarly queries with OpenAlex. "
            "Each stage is bounded; no scientific novelty is claimed."
        )
        key = configured_api_key()
        if not key:
            st.info("Configure GROQ_API_KEY in .env to enable the coordinated AI workflow.")
        elif st.button("Run full multi-agent workflow", disabled=not batch.ready_for_analysis):
            with st.spinner("Running all four agents and checking literature evidence..."):
                try:
                    model = MeteredLLM(get_app_llm())
                    search = MeteredSearch(OpenAlexClient(api_key=os.getenv("OPENALEX_API_KEY", "").strip() or None))
                    flow = ResearchWorkflow(
                        PaperAnalyzer(model), PaperComparator(model), GapDiscoverer(model),
                        GapVerifier(model, search),
                    )
                    began = perf_counter()
                    result = flow.run(batch)
                    underlying = getattr(model, "delegate", None)
                    if hasattr(underlying, "rotation_history") and underlying.rotation_history:
                        for event in underlying.rotation_history:
                            st.info(
                                f"🔄 Key rotation: shifted from {event['from_provider'].title()} (Key {event['from_key']}) "
                                f"to {event['to_provider'].title()} (Key {event['to_key']}) due to {event['reason']}."
                            )
                    st.session_state.workflow_telemetry = {
                        "workflow_wall_seconds": round(perf_counter() - began, 4),
                        "llm": model.measurements(), "literature_search": search.measurements(),
                    }
                    st.session_state.workflow_result = result
                    # Existing manual stage views can inspect the SAME source-linked
                    # results on subsequent Streamlit reruns without duplicate API calls.
                    for key_name, obj in (
                        ("analysis_result", result.paper_analyses),
                        ("comparison_result", result.paper_comparison),
                        ("gap_result", result.research_gaps),
                        ("verification_result", result.literature_verification),
                    ):
                        if obj is None:
                            st.session_state.pop(key_name, None)
                        else:
                            st.session_state[key_name] = obj
                except WorkflowDependencyError as exc:
                    st.error(str(exc))
                except (ValueError, RuntimeError) as exc:
                    st.error(f"Unable to start the full workflow: {type(exc).__name__}. Check configuration and logs.")

        workflow_result = st.session_state.get("workflow_result")
        if workflow_result is not None:
            a, b, c = st.columns(3)
            a.metric("Workflow status", workflow_result.status)
            b.metric("Revisions attempted", workflow_result.revision_count)
            c.metric("Literature assessments", (
                len(workflow_result.literature_verification.verifications)
                if workflow_result.literature_verification else 0
            ))
            if workflow_result.blocked_reason:
                st.warning(workflow_result.blocked_reason)
            st.write("**Execution trace**")
            for event in workflow_result.events:
                st.write(f"• {event.stage.title()} [{event.status}]: {event.message}")
            if workflow_result.revision_attempted:
                st.info(
                    "A challenged gap triggered one bounded revision. "
                    + ("Updated candidates were searched again." if workflow_result.revision_applied
                       else "No valid replacement was found; original verification was retained.")
                )
            if workflow_result.literature_verification:
                st.write("**Final gap verification verdicts**")
                for item in workflow_result.literature_verification.verifications:
                    st.write(f"• {item.gap_id}: {item.title} — {item.status.replace('_', ' ')}")
            st.warning("Research novelty is never established automatically; human review is required.")
            st.download_button(
                "Download complete LangGraph workflow (JSON)",
                data=workflow_result.model_dump_json(indent=2).encode("utf-8"),
                file_name="researchgap_workflow_report.json", mime="application/json",
            )




with workspace_tabs[6]:
    st.subheader("✦ Phase 7 — Research Report Studio")
    st.caption("Create an academic portfolio-ready literature review, with complete source evidence and transparent verification limitations.")
    render_report_studio(st)

with workspace_tabs[7]:
    st.subheader("Phase 8 — Evaluation & reproducibility")
    render_evaluation_studio(st)

st.markdown(
    '<div class="rg-footnote">ResearchGap AI is a research-support tool, not a novelty certification service. '
    'Cross-paper claims, source excerpts, and external literature checks require independent human review.</div>',
    unsafe_allow_html=True,
)
