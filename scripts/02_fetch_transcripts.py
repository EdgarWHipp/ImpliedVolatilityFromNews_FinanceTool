"""Stage 2 batch fetch: walk universe x 4 quarters, cache transcripts to disk,
print a coverage table.

Default window: Q1-Q4 2023.  Override with --year-quarters "2023:1,2023:2,...".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iv_news.data.transcripts import get_transcript
from iv_news.data.universe import tickers as universe_tickers


DEFAULT_YQ = [(2023, 1), (2023, 2), (2023, 3), (2023, 4)]

STATUS_GLYPH = {
    "transcript": "T",
    "press_release": "P",
    "other": "?",
    "no_filing": "-",
    "no_exhibit": "x",
}


def parse_yq(arg: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in arg.split(","):
        y, q = tok.split(":")
        out.append((int(y), int(q)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year-quarters", type=parse_yq, default=DEFAULT_YQ)
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Subset of universe to fetch (default: all 50).")
    parser.add_argument("--force", action="store_true",
                        help="Bypass on-disk cache.")
    args = parser.parse_args()

    tks = args.tickers or universe_tickers()
    yqs = args.year_quarters
    print(f"Fetching {len(tks)} tickers × {len(yqs)} quarters = {len(tks) * len(yqs)} cells")
    print(f"Window: {yqs}")
    print()

    # Header
    yq_labels = [f"{y}Q{q}" for y, q in yqs]
    col_w = max(len(s) for s in yq_labels) + 1
    header = "TICKER  " + " ".join(s.rjust(col_w) for s in yq_labels)
    print(header)
    print("-" * len(header))

    counts: dict[str, int] = {}
    for tk in tks:
        cells = []
        for y, q in yqs:
            try:
                r = get_transcript(tk, y, q, force=args.force)
                glyph = STATUS_GLYPH.get(r.status, "?")
            except Exception as e:  # noqa: BLE001
                glyph = "E"
                print(f"  ! {tk} {y}Q{q}: {type(e).__name__}: {e}", file=sys.stderr)
            counts[glyph] = counts.get(glyph, 0) + 1
            cells.append(glyph)
        print(f"{tk:<7} " + " ".join(c.rjust(col_w) for c in cells))

    print()
    print("Legend: T=transcript  P=press_release  ?=other  -=no_filing  "
          "x=no_exhibit  E=error")
    print("Counts:", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
