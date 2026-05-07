from __future__ import annotations

from datetime import date

import pytest

from iv_news.data import edgar


def test_quarter_search_window_q1():
    start, end = edgar.quarter_search_window(2023, 1)
    # Q1 ends 2023-03-31; window is +5 to +100 days.
    assert start == date(2023, 4, 5)
    assert end == date(2023, 7, 9)


def test_quarter_search_window_q4_year_rollover():
    start, end = edgar.quarter_search_window(2023, 4)
    assert start == date(2024, 1, 5)
    assert end == date(2024, 4, 9)


def test_quarter_search_window_invalid():
    with pytest.raises(ValueError):
        edgar.quarter_search_window(2023, 5)


def test_filing_url_helpers():
    f = edgar.Filing(
        cik="0000320193",
        ticker="AAPL",
        accession="0000320193-24-000123",
        filing_date=date(2024, 11, 1),
        form="8-K",
        primary_doc="aapl-20241101.htm",
        items=("2.02", "9.01"),
    )
    assert f.accession_nodash == "000032019324000123"
    assert f.archive_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123"
    )
    assert f.index_json_url.endswith("/index.json")


def test_parse_recent_filings_filters():
    submissions = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-1", "0001-2", "0001-3"],
                "filingDate": ["2023-04-30", "2023-08-01", "2023-08-02"],
                "form": ["8-K", "10-Q", "8-K"],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
                "items": ["2.02,9.01", "", "5.02"],
            }
        }
    }
    parsed = edgar._parse_recent_filings(submissions, cik="0000000001", ticker="X")
    assert len(parsed) == 3
    # Items parsing
    assert parsed[0].items == ("2.02", "9.01")
    assert parsed[1].items == ()
    assert parsed[2].items == ("5.02",)
