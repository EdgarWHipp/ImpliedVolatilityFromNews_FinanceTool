from __future__ import annotations

from iv_news.data.text_extract import classify, html_to_text


def test_html_to_text_strips_tags_and_normalizes_whitespace():
    html = b"""
    <html><head><style>.x{color:red}</style></head><body>
      <h1>  Title   </h1>
      <p>First   paragraph.</p>
      <script>alert('x')</script>
      <p>Second paragraph.\n\n\nWith   extra   space.</p>
    </body></html>
    """
    text = html_to_text(html)
    assert "Title" in text
    assert "First paragraph." in text
    assert "Second paragraph." in text
    assert "alert" not in text  # script stripped
    assert "color:red" not in text  # style stripped
    # No runs of 3+ newlines or 2+ spaces.
    assert "\n\n\n" not in text
    assert "  " not in text


def test_classify_transcript():
    body = (
        "Conference Call Transcript\n\n"
        "Operator: Good morning and welcome to the Q3 earnings call.\n\n"
        "Thank you, operator. We are pleased to share our prepared remarks today.\n\n"
        + ("Strong revenue growth in the quarter. " * 600)
        + "\n\nQuestion-and-Answer Session\n\nOperator: First question.\n"
    )
    res = classify(body)
    assert res.doc_class == "transcript", res
    assert res.transcript_markers >= 2
    assert res.word_count > 3000


def test_classify_press_release():
    body = (
        "FOR IMMEDIATE RELEASE\n\nNews Release\n\n"
        "Company X reports record results.\n\n"
        + ("Revenue grew. " * 500)
        + "\n\nNon-GAAP financial measures are presented in this release."
    )
    res = classify(body)
    assert res.doc_class == "press_release", res
    assert res.press_release_markers >= 1


def test_classify_too_short_is_other():
    res = classify("Just a few words here.")
    assert res.doc_class == "other"
    assert res.reason == "too short"


def test_classify_long_but_no_markers_is_other():
    body = "Some boring tabular content. " * 500
    res = classify(body)
    assert res.doc_class == "other"
