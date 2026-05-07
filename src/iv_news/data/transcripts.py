"""Transcript orchestrator: cache-aware (ticker, year, quarter) -> text on disk.

Layout:
    data/transcripts/{TICKER}/{YYYY}_Q{N}.txt        # cleaned text
    data/transcripts/{TICKER}/{YYYY}_Q{N}.meta.json  # source metadata

A meta sidecar is written even on misses, so a re-run skips re-checking
EDGAR for cells that previously yielded nothing. Pass force=True to bypass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from iv_news.config import TRANSCRIPTS_DIR
from iv_news.data import edgar
from iv_news.data.text_extract import ClassifyResult, classify, html_to_text


@dataclass
class TranscriptResult:
    ticker: str
    year: int
    quarter: int
    status: str  # "transcript" | "press_release" | "other" | "no_filing" | "no_exhibit"
    text_path: Optional[str] = None
    word_count: int = 0
    source_url: Optional[str] = None
    accession: Optional[str] = None
    filing_date: Optional[str] = None
    reason: Optional[str] = None


def _cell_paths(ticker: str, year: int, quarter: int) -> tuple[Path, Path]:
    base = TRANSCRIPTS_DIR / ticker.upper()
    txt = base / f"{year}_Q{quarter}.txt"
    meta = base / f"{year}_Q{quarter}.meta.json"
    return txt, meta


def _save(result: TranscriptResult, text: Optional[str]) -> None:
    txt_path, meta_path = _cell_paths(result.ticker, result.year, result.quarter)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if text is not None:
        txt_path.write_text(text, encoding="utf-8")
        result.text_path = str(txt_path)
    meta_path.write_text(json.dumps(asdict(result), indent=2))


def _load_cached(ticker: str, year: int, quarter: int) -> Optional[TranscriptResult]:
    _, meta_path = _cell_paths(ticker, year, quarter)
    if not meta_path.exists():
        return None
    raw = json.loads(meta_path.read_text())
    return TranscriptResult(**raw)


def get_transcript(
    ticker: str, year: int, quarter: int, force: bool = False
) -> TranscriptResult:
    """Resolve and cache the transcript (or whatever is filed) for one cell."""
    if not force:
        cached = _load_cached(ticker, year, quarter)
        if cached is not None:
            return cached

    cik = edgar.cik_for_ticker(ticker)
    filings = edgar.list_earnings_8ks(cik, year, quarter, ticker=ticker)
    if not filings:
        result = TranscriptResult(
            ticker=ticker,
            year=year,
            quarter=quarter,
            status="no_filing",
            reason="no 8-K with item 2.02/7.01 in window",
        )
        _save(result, None)
        return result

    # Try each candidate, prefer the largest exhibit across all of them.
    best: Optional[tuple[ClassifyResult, str, edgar.Filing, dict, str]] = None
    for filing in filings:
        fetched = edgar.fetch_exhibit_99(filing)
        if fetched is None:
            continue
        raw, url, doc = fetched
        text = html_to_text(raw)
        cr = classify(text)
        rank_score = (
            {"transcript": 3, "press_release": 2, "other": 1}[cr.doc_class],
            cr.word_count,
        )
        if best is None or rank_score > (
            {"transcript": 3, "press_release": 2, "other": 1}[best[0].doc_class],
            best[0].word_count,
        ):
            best = (cr, text, filing, doc, url)

    if best is None:
        f = filings[0]
        result = TranscriptResult(
            ticker=ticker,
            year=year,
            quarter=quarter,
            status="no_exhibit",
            accession=f.accession,
            filing_date=str(f.filing_date),
            reason="no EX-99 found in any candidate filing",
        )
        _save(result, None)
        return result

    cr, text, filing, doc, url = best
    result = TranscriptResult(
        ticker=ticker,
        year=year,
        quarter=quarter,
        status=cr.doc_class,
        word_count=cr.word_count,
        source_url=url,
        accession=filing.accession,
        filing_date=str(filing.filing_date),
        reason=cr.reason,
    )
    _save(result, text)
    return result
