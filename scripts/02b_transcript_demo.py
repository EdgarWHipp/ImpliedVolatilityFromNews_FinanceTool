"""Stage 2 live demo: fetch a small slice (3 tickers x 1 quarter) so you can
eyeball the pipeline end-to-end before scaling to the full universe.

Run:
    python scripts/02b_transcript_demo.py
    python scripts/02b_transcript_demo.py --tickers AAPL MSFT --year 2023 --quarter 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iv_news.data.transcripts import get_transcript

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM"]
DEFAULT_YEAR = 2023
DEFAULT_QUARTER = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR)
    parser.add_argument("--quarter", type=int, default=DEFAULT_QUARTER)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"Demo: fetching {args.tickers} for {args.year} Q{args.quarter}\n")
    for ticker in args.tickers:
        print(f"=== {ticker} ===")
        r = get_transcript(ticker, args.year, args.quarter, force=args.force)
        print(f"  status:       {r.status}")
        print(f"  word_count:   {r.word_count}")
        print(f"  filing_date:  {r.filing_date}")
        print(f"  accession:    {r.accession}")
        print(f"  source_url:   {r.source_url}")
        print(f"  reason:       {r.reason}")
        if r.text_path:
            text = Path(r.text_path).read_text(encoding="utf-8")
            print(f"  text_path:    {r.text_path}  ({len(text):,} chars)")
            preview = text[:500].replace("\n", " ")
            print(f"  preview:      {preview}...")
        print()


if __name__ == "__main__":
    main()
