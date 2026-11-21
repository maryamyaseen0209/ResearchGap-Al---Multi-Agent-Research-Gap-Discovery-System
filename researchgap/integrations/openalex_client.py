"""OpenAlex works search with retries, safe responses and no fabricated metadata.

Uses the current /works `search.title_abstract_keywords` parameter. Requests are
sent only after the user presses the Phase 5 button or explicitly runs its CLI.
"""

from dataclasses import dataclass
import re
import time

import requests

from researchgap.core.verification_schemas import LiteratureWork

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
WORK_ID = re.compile(r"(?:https://(?:www\.)?openalex\.org/)?(W[0-9]+)$")
SELECT_FIELDS = (
    "id,title,display_name,publication_year,doi,abstract_inverted_index,cited_by_count"
)


class LiteratureSearchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def reconstruct_abstract(index: dict | None) -> str | None:
    """OpenAlex stores abstracts as word -> positions, not directly as text."""
    if not index or not isinstance(index, dict):
        return None
    positions: dict[int, str] = {}
    for word, offsets in index.items():
        if not isinstance(word, str) or not isinstance(offsets, list):
            continue
        for offset in offsets:
            if type(offset) is int and 0 <= offset <= 10000 and offset not in positions:
                positions[offset] = word
    if not positions or max(positions) > max(len(positions) * 3, 200):
        return None
    return " ".join(positions[i] for i in sorted(positions))[:16000].strip() or None


def parse_work(raw: object) -> LiteratureWork | None:
    if not isinstance(raw, dict):
        return None
    match = WORK_ID.fullmatch(str(raw.get("id") or ""))
    if not match:
        return None
    work_id = match.group(1)
    title = raw.get("title") or raw.get("display_name")
    if not isinstance(title, str) or not title.strip():
        return None
    doi = raw.get("doi")
    if not (isinstance(doi, str) and re.fullmatch(r"https://doi\.org/10\.[^\s]{3,250}", doi)):
        doi = None
    year = raw.get("publication_year")
    if type(year) is not int or not 1800 <= year <= 2200:
        year = None
    citations = raw.get("cited_by_count")
    if type(citations) is not int or citations < 0:
        citations = None
    return LiteratureWork(
        openalex_id=work_id, title=title.strip()[:500],
        url=f"https://openalex.org/{work_id}", publication_year=year, doi=doi,
        abstract=reconstruct_abstract(raw.get("abstract_inverted_index")),
        cited_by_count=citations,
    )


@dataclass
class OpenAlexClient:
    """Search tool. Keyless calls work for small demos; free key is recommended."""

    api_key: str | None = None
    timeout_seconds: float = 15
    max_retries: int = 2
    session: requests.Session | None = None

    def __post_init__(self):
        if not 2 <= self.timeout_seconds <= 60:
            raise ValueError("timeout_seconds must be 2–60")
        if not 0 <= self.max_retries <= 4:
            raise ValueError("max_retries must be 0–4")
        if self.session is None:
            self.session = requests.Session()

    def search(self, query: str, limit: int = 5) -> list[LiteratureWork]:
        if not query.strip() or len(query) > 180:
            raise ValueError("Search query must be between 1 and 180 characters")
        if not 1 <= limit <= 15:
            raise ValueError("Search result limit must be between 1 and 15")
        params = {
            "search.title_abstract_keywords": query,
            "per_page": limit,
            "select": SELECT_FIELDS,
        }
        if self.api_key and self.api_key.strip():
            params["api_key"] = self.api_key.strip()
        assert self.session is not None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(
                    OPENALEX_WORKS_URL, params=params,
                    headers={"Accept": "application/json", "User-Agent": "ResearchGapAI/0.5"},
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 3))
                    continue
                raise LiteratureSearchError("NETWORK_ERROR", "Could not connect to OpenAlex.") from exc
            if response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if re.fullmatch(r"\d+(?:\.\d+)?", retry_after) else 2 ** attempt
                time.sleep(min(delay, 5))
                continue
            if response.status_code in (401, 403):
                raise LiteratureSearchError("AUTH_ERROR", "OpenAlex authentication was rejected.")
            if response.status_code == 429:
                raise LiteratureSearchError("RATE_LIMITED", "OpenAlex quota or request limit reached.")
            if response.status_code >= 400:
                raise LiteratureSearchError("API_ERROR", f"OpenAlex returned HTTP {response.status_code}.")
            try:
                data = response.json()
                if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                    raise ValueError("Expected OpenAlex results list")
                results = []
                seen = set()
                records = data["results"][:limit]
                for raw in records:
                    work = parse_work(raw)
                    if work and work.openalex_id not in seen:
                        results.append(work)
                        seen.add(work.openalex_id)
                if records and not results:
                    raise LiteratureSearchError(
                        "BAD_RESPONSE", "OpenAlex results contained no usable work records."
                    )
                return results
            except (ValueError, TypeError) as exc:
                raise LiteratureSearchError("BAD_RESPONSE", "OpenAlex returned invalid results data.") from exc
        raise LiteratureSearchError("API_ERROR", "OpenAlex request failed after retries.")
