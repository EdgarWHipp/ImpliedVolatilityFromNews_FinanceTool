"""HTML -> clean text and transcript-vs-press-release heuristic classifier.

We treat exhibits as one of:
  - transcript     : prepared remarks + Q&A, what we actually want
  - press_release  : numbers + bullet quotes, lower-signal but still usable
  - other          : anything else (presentation slides, financial tables only)

The heuristic is intentionally simple: look for transcript-shaped markers
(Operator, Q&A, named speakers) and word-count gates. Stage 3's LLM extractor
handles all three classes; the label just lets us weight confidence later.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Literal

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Some SEC exhibits are technically XML but render as HTML; we don't care
# either way and BS4's html parser handles both. Silence the spurious warning.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DocClass = Literal["transcript", "press_release", "other"]

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def html_to_text(html: bytes | str) -> str:
    """Strip tags, normalize whitespace, preserve paragraph breaks."""
    if isinstance(html, bytes):
        # Most SEC exhibits are utf-8; fall back to latin-1 if not.
        try:
            html = html.decode("utf-8")
        except UnicodeDecodeError:
            html = html.decode("latin-1", errors="replace")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # get_text with newlines between block elements
    text = soup.get_text(separator="\n")
    # Collapse runs of spaces, then runs of blank lines.
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    # Strip leading/trailing whitespace per line.
    text = "\n".join(line.strip() for line in text.splitlines())
    return text.strip()


# Markers that strongly suggest a real conference-call transcript.
_TRANSCRIPT_MARKERS = [
    re.compile(r"\bconference call\b", re.I),
    re.compile(r"\bearnings call\b", re.I),
    re.compile(r"\bprepared remarks\b", re.I),
    re.compile(r"^operator[:\s]", re.I | re.M),
    re.compile(r"\bquestion[- ]and[- ]answer\b", re.I),
    re.compile(r"\bQ\s*&\s*A\b", re.I),
    re.compile(r"^thank you,? operator\b", re.I | re.M),
]

# Markers that suggest "this is just the press release / earnings release".
# Includes the formal boilerplate (FOR IMMEDIATE RELEASE, etc.) plus the
# substantive markers used by companies like Amazon that skip the boilerplate
# but include the canonical exhibit/exchange/quarter framing.
_PRESS_RELEASE_MARKERS = [
    re.compile(r"\bnews release\b", re.I),
    re.compile(r"\bpress release\b", re.I),
    re.compile(r"\bfor immediate release\b", re.I),
    re.compile(r"\bnon-?GAAP financial measures?\b", re.I),
    re.compile(r"\bexhibit\s*99(?:\.\d)?\b", re.I),
    re.compile(r"\bbusiness wire\b", re.I),
    re.compile(r"\(\s*NASDAQ\s*:\s*[A-Z\.]{1,6}\s*\)", re.I),
    re.compile(r"\(\s*NYSE\s*:\s*[A-Z\.]{1,6}\s*\)", re.I),
    re.compile(r"\b(announces|announced|reports|reported)\b.{0,40}\bquarter\b", re.I),
    re.compile(
        r"\bended\s+(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\b",
        re.I,
    ),
    re.compile(r"\btoday announced\b", re.I),
    re.compile(r"\bearnings release\b", re.I),
]


@dataclass(frozen=True)
class ClassifyResult:
    doc_class: DocClass
    word_count: int
    transcript_markers: int
    press_release_markers: int
    reason: str


def classify(text: str) -> ClassifyResult:
    """Decide whether `text` is a transcript, press release, or other.

    Rules (applied in order):
      1. If word count < 800, it's "other" (likely just a cover page or table).
      2. If >=2 transcript markers and word count > 3000, it's a transcript.
      3. If any press-release markers, it's a press_release.
      4. Otherwise, it's "other".
    """
    wc = len(text.split())
    t_hits = sum(1 for r in _TRANSCRIPT_MARKERS if r.search(text))
    p_hits = sum(1 for r in _PRESS_RELEASE_MARKERS if r.search(text))

    if wc < 800:
        return ClassifyResult("other", wc, t_hits, p_hits, "too short")
    if t_hits >= 2 and wc > 3000:
        return ClassifyResult(
            "transcript", wc, t_hits, p_hits, f"{t_hits} transcript markers, wc={wc}"
        )
    if p_hits >= 1:
        return ClassifyResult(
            "press_release", wc, t_hits, p_hits, f"{p_hits} press-release markers"
        )
    return ClassifyResult("other", wc, t_hits, p_hits, "no decisive markers")
