# ImpliedVolatilityFromNews_FinanceTool

Blending RMT-cleaned covariance with LLM-extracted relationship priors from earnings call transcripts.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env  # fill in FMP_API_KEY
```

## Stage 1 — baseline covariance

```bash
python scripts/00_build_universe.py     # sanity-check universe.csv
python scripts/01_baseline_cov.py       # fetch returns, compute sample + RMT cov
pytest                                   # synthetic factor tests
```
