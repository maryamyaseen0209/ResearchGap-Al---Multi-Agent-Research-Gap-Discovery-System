# Phase 8 — Deployment and Release Checklist

## Supported deployment: Streamlit Community Cloud

1. Push the **ResearchGap-AI** folder contents to your GitHub repository (with `app.py`, `requirements.txt`, and `.streamlit/config.toml` at the repository root).
2. Make sure the GitHub Actions workflow `.github/workflows/ci.yml` is green. CI installs **real LangGraph and Streamlit**, checks the UI, runs all tests and executes the offline complete pipeline with a real LangGraph runtime.
3. In [Streamlit Community Cloud](https://share.streamlit.io/), create a new app from that repository. Set the main file path to `app.py` and choose **Python 3.12** for consistency with CI.
4. Under *Advanced settings / Secrets*, configure top-level keys as in `.streamlit/secrets.toml.example`. **Never commit `.env` or `secrets.toml`.** Root-level secrets are exposed to the app as environment variables by Streamlit's secrets manager.
5. Deploy, inspect logs, upload three non-sensitive PDFs, and manually verify the entire workflow with your own API keys and accessible OpenAlex service. Validate actual agent outputs, report citation references and at least one incorrect citation rejection.
6. Download Markdown/HTML/JSON, confirm HTML opens, review the evaluation metrics, and check the application on desktop and mobile widths.

## Security and usage cautions

- A public deployment with the `GROQ_API_KEY` configured lets **visitors incur charges and consume quotas through your account**. Restrict access and usage or keep the app private until you implement production authentication, per-user rate limits, and spending controls. Streamlit's per-session state is not cross-user rate limiting.
- Uploaded papers and selected excerpts leave the app only when a user explicitly runs an AI operation; content then goes to Groq. Verification queries go to OpenAlex. Users should have permission to upload those documents. Do not use confidential papers without review of provider terms.
- The current app uses in-memory session state and LangGraph's memory saver. It does not provide persistent task storage, durable review history, authentication or tenant isolation.
- `No direct match found` does not mean a topic is new. Keep all novelty and human-review disclaimers visible.
- API quotas/pricing/model availability may change. Inspect Groq and OpenAlex accounts before enabling real usage.

## Deployment verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m examples.demo_phase8_offline --require-langgraph --output-dir evaluation_output/phase8-ci
python -m streamlit run app.py --server.headless true
```

Then check `http://localhost:8501` in a real browser. Automated AppTest tests UI layout and the CI job checks the Streamlit health endpoint, but **neither substitutes for interactive browser/mobile review**.

## What cannot be certified from the offline build environment

Groq credentials and external OpenAlex connectivity were not available, and package installation for Streamlit/LangGraph was blocked. The installed dependencies and network access of your Cloud environment must be verified after deployment; until that occurs, the project is **deploy-ready**, not **live-deployed**.
