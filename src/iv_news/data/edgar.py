"""SEC EDGAR client: ticker -> CIK, list earnings 8-Ks, fetch Exhibit 99.

SEC requires a contact-info User-Agent on every request and asks for at
most 10 req/sec across the whole API. We hardcode 5/sec via a tiny token
bucket and a shared httpx client. All network calls are wrapped in a
retry decorator that respects 429 / 5xx with exponential backoff.

Public surface:
    cik_for_ticker(ticker)        -> "0000320193"
    list_earnings_8ks(cik, year, quarter) -> list[Filing]
    fetch_exhibit_99(filing)      -> tuple[bytes, str]   # raw html bytes + url
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Iterable

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from iv_news.config import CACHE_DIR, SEC_CALLS_PER_SEC, SEC_USER_AGENT

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}"

EARNINGS_ITEMS = {"2.02", "7.01"}  # Results of Operations / Reg FD

# The `type` field in EDGAR's index.json is unreliable (often returns icon
# filenames like "text.gif"). Companies also use wildly varying exhibit
# filename conventions (e.g. NVDA's `q3fy24pr.htm`, AMD's `q32023991.htm`,
# META's `meta09302023-exhibit991.htm`). Rather than chase patterns, we
# pick the largest non-skipped HTML document in the filing -- that is
# reliably the press release / financial supplement / transcript.
_SKIP_NAME_RE = re.compile(
    r"\.(?:xml|xsd|zip|jpg|jpeg|png|gif|css|js|xlsx|json)$"
    r"|^FilingSummary"
    r"|^MetaLinks"
    r"|-index|-index-headers|^R\d+\.htm$",
    re.IGNORECASE,
)
# Reject anything that is not text/HTML.
_TEXT_DOC_EXT = (".htm", ".html", ".txt")


# --------------------------------------------------------------------------- #
# Rate-limited HTTP client
# --------------------------------------------------------------------------- #

_last_call_lock = threading.Lock()
_last_call_at = 0.0
_min_interval = 1.0 / SEC_CALLS_PER_SEC


def _throttle() -> None:
    """Block until at least _min_interval has passed since the last call."""
    global _last_call_at
    with _last_call_lock:
        now = time.monotonic()
        wait = _min_interval - (now - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


@lru_cache(maxsize=1)
def _client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": SEC_USER_AGENT,
            "Accept-Encoding": "gzip, deflate",
        },
        timeout=30.0,
        follow_redirects=True,
    )


class TransientHTTPError(Exception):
    pass


@retry(
    retry=retry_if_exception_type((TransientHTTPError, httpx.TransportError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1.5, min=1, max=20),
    reraise=True,
)
def _get(url: str) -> httpx.Response:
    _throttle()
    resp = _client().get(url)
    if resp.status_code in (429, 500, 502, 503, 504):
        raise TransientHTTPError(f"{resp.status_code} on {url}")
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------- #
# Ticker -> CIK
# --------------------------------------------------------------------------- #

_TICKER_CACHE = CACHE_DIR / "sec_company_tickers.json"


def _load_ticker_map() -> dict[str, str]:
    """Return ticker (upper) -> 10-digit CIK string."""
    if _TICKER_CACHE.exists():
        raw = json.loads(_TICKER_CACHE.read_text())
    else:
        resp = _get(TICKERS_URL)
        raw = resp.json()
        _TICKER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _TICKER_CACHE.write_text(json.dumps(raw))
    out: dict[str, str] = {}
    for entry in raw.values():
        ticker = str(entry["ticker"]).upper()
        cik10 = f"{int(entry['cik_str']):010d}"
        out[ticker] = cik10
    return out


@lru_cache(maxsize=1)
def _ticker_map() -> dict[str, str]:
    return _load_ticker_map()


def cik_for_ticker(ticker: str) -> str:
    m = _ticker_map()
    key = ticker.upper()
    if key not in m:
        raise KeyError(f"Ticker {ticker!r} not found in SEC ticker map.")
    return m[key]


# --------------------------------------------------------------------------- #
# Quarter windows
# --------------------------------------------------------------------------- #

def quarter_search_window(year: int, quarter: int) -> tuple[date, date]:
    """Filing date window in which the (year, quarter) earnings 8-K is expected.

    Banks (e.g. JPM, WFC) report as early as ~13 days after period end,
    while slow-reporters fall in the 75-90 day range. We use
    [period_end + 5, period_end + 100] to be inclusive on both ends.
    """
    if quarter not in {1, 2, 3, 4}:
        raise ValueError(f"quarter must be 1..4, got {quarter}")
    end_month = quarter * 3
    last_day = {3: 31, 6: 30, 9: 30, 12: 31}[end_month]
    period_end = date(year, end_month, last_day)
    start = period_end + timedelta(days=5)
    end = period_end + timedelta(days=100)
    return start, end


# --------------------------------------------------------------------------- #
# Filings
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Filing:
    cik: str           # 10-digit
    ticker: str
    accession: str     # e.g. "0000320193-24-000123"
    filing_date: date
    form: str          # "8-K"
    primary_doc: str   # e.g. "aapl-20240131.htm"
    items: tuple[str, ...]

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def archive_url(self) -> str:
        return ARCHIVE_BASE.format(
            cik_int=int(self.cik), accession_nodash=self.accession_nodash
        )

    @property
    def index_json_url(self) -> str:
        return f"{self.archive_url}/index.json"


def _parse_filings_block(block: dict, cik: str, ticker: str) -> list[Filing]:
    """Parse the parallel-array block used by both the `recent` slot and
    each older submission shard."""
    n = len(block.get("accessionNumber", []))
    out: list[Filing] = []
    for i in range(n):
        items_raw = block["items"][i] if "items" in block else ""
        items = tuple(s.strip() for s in items_raw.split(",") if s.strip())
        out.append(
            Filing(
                cik=cik,
                ticker=ticker,
                accession=block["accessionNumber"][i],
                filing_date=date.fromisoformat(block["filingDate"][i]),
                form=block["form"][i],
                primary_doc=block["primaryDocument"][i],
                items=items,
            )
        )
    return out


# Backwards-compatible alias used in tests.
_parse_recent_filings = lambda submissions, cik, ticker: _parse_filings_block(  # noqa: E731
    submissions["filings"]["recent"], cik=cik, ticker=ticker
)


def _all_filings_in_window(
    cik: str, ticker: str, start: date, end: date
) -> list[Filing]:
    """Fetch the recent block plus any older submission shard whose date
    range overlaps [start, end]. High-volume filers (banks, ETF issuers)
    need the older shards because `recent` only holds ~1000 entries."""
    url = SUBMISSIONS_URL.format(cik10=cik)
    submissions = _get(url).json()
    out = _parse_filings_block(
        submissions["filings"]["recent"], cik=cik, ticker=ticker
    )
    for entry in submissions["filings"].get("files", []):
        f_from = date.fromisoformat(entry["filingFrom"])
        f_to = date.fromisoformat(entry["filingTo"])
        # Overlap test: [f_from, f_to] intersects [start, end]
        if f_to < start or f_from > end:
            continue
        shard_url = f"https://data.sec.gov/submissions/{entry['name']}"
        shard = _get(shard_url).json()
        out.extend(_parse_filings_block(shard, cik=cik, ticker=ticker))
    return out


def list_earnings_8ks(cik: str, year: int, quarter: int, ticker: str) -> list[Filing]:
    """8-K filings whose date is in the quarter window AND that include an
    earnings-related item (2.02 or 7.01)."""
    start, end = quarter_search_window(year, quarter)
    filings = _all_filings_in_window(cik, ticker, start, end)
    return [
        f
        for f in filings
        if f.form == "8-K"
        and start <= f.filing_date <= end
        and (set(f.items) & EARNINGS_ITEMS)
    ]


def list_filing_documents(filing: Filing) -> list[dict]:
    """Returns list of {name, type, size, ...} documents in a filing."""
    data = _get(filing.index_json_url).json()
    return data["directory"]["item"]


def find_largest_ex99(filing: Filing) -> dict | None:
    """Pick the largest text document in the filing as the press release.

    Rationale: across our universe, exhibit filenames vary too much for
    any pattern-matching approach to be reliable. But within an earnings
    8-K the press release / Ex-99 supplement is consistently the largest
    .htm/.html/.txt document. The 8-K cover, if present, is much smaller.
    Pure-image filings (e.g. F, where exhibits are JPEGs of slides) yield
    no candidate and return None.
    """
    docs = list_filing_documents(filing)
    candidates: list[dict] = []
    for d in docs:
        name = str(d.get("name", ""))
        if _SKIP_NAME_RE.search(name):
            continue
        if not name.lower().endswith(_TEXT_DOC_EXT):
            continue
        candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda d: int(d.get("size") or 0), reverse=True)
    return candidates[0]


def fetch_exhibit_99(filing: Filing) -> tuple[bytes, str, dict] | None:
    """Returns (raw bytes, url, document dict) for the largest Ex-99, or None."""
    doc = find_largest_ex99(filing)
    if doc is None:
        return None
    url = f"{filing.archive_url}/{doc['name']}"
    resp = _get(url)
    return resp.content, url, doc


# --------------------------------------------------------------------------- #
# Convenience iterator
# --------------------------------------------------------------------------- #

def iter_earnings_filings(
    tickers: Iterable[str], year_quarters: Iterable[tuple[int, int]]
) -> Iterable[tuple[str, int, int, list[Filing]]]:
    """Yield (ticker, year, quarter, filings) for each cell in the grid."""
    yqs = list(year_quarters)
    for t in tickers:
        cik = cik_for_ticker(t)
        for y, q in yqs:
            yield t, y, q, list_earnings_8ks(cik, y, q, ticker=t)
