from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
CACHE_DIR = DATA_DIR / "cache"

load_dotenv(REPO_ROOT / ".env")

FMP_API_KEY = os.getenv("FMP_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# FMP free tier: 250 calls/day, no transcript endpoint.
# Self-imposed pacing to be polite even with paid plans.
FMP_CALLS_PER_SEC = 5

# SEC EDGAR requires "Name email" in the User-Agent. SEC asks ≤10 req/sec;
# we cap at 5/sec to be polite.
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "Edgar Hipp klavierboy4.hipp@gmail.com",
)
SEC_CALLS_PER_SEC = 5
