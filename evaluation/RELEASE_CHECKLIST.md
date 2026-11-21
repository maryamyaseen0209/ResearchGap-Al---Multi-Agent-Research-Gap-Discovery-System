# ResearchGap AI — Phase 8 Release Checklist

- [x] Latest Phase 1–7 code preserved in same project tree
- [x] PDF upload, extraction, page offsets and duplicates tested
- [x] Four agent stages tested with fictional source PDFs and simulated API replies
- [x] Conditional workflow routing and bounded revision tested with graph simulator
- [x] PDF citation integrity rechecked in end-to-end smoke
- [x] Baseline (shared Phase 2 facts, single-pass) implemented
- [x] Automated citation and source coverage metrics implemented
- [x] Human annotation CSV template and scoring CLI implemented
- [x] Model-call/latency instrumentation and optional provider token usage implemented
- [x] Markdown, HTML, JSON report generation tested
- [x] Eight-tab UI mocked and regression-tested
- [x] CI workflow, Streamlit configuration and deployment guide added
- [ ] Real LangGraph runtime executed here (blocked: cannot install package in build environment)
- [ ] Live Streamlit browser/mobile rendering visually reviewed (blocked: Streamlit unavailable)
- [ ] Groq real model and key tested (requires user credentials)
- [ ] OpenAlex internet request verified from deployment target
- [ ] Real-paper scientific accuracy evaluated and independently annotated
- [ ] GitHub Actions CI actually run on user's repository
- [ ] Public deployment and rate limiting/user access restrictions established

**Quality gate:** The code is ready for local/cloud validation, but real deployment, API functionality, and scientific reliability remain unverified until the unchecked items are completed.
