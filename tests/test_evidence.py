"""Tests for evidence retrieval and prompt packaging."""

from hermes_council.evidence import (
    EvidenceBundle,
    EvidenceSource,
    collect_evidence,
    extract_urls,
)


def test_extract_urls_dedupes_and_strips_punctuation():
    text = "Read https://example.com/a, then https://example.com/a and https://b.example/x."
    assert extract_urls(text) == ["https://example.com/a", "https://b.example/x"]


def test_collect_evidence_fetches_provided_urls(monkeypatch):
    def fake_read_url(url, timeout):
        assert url == "https://example.com/report"
        return "<html><title>Report</title><body>Important source text about the claim.</body></html>"

    monkeypatch.setenv("COUNCIL_EVIDENCE_SEARCH", "0")
    monkeypatch.setattr("hermes_council.evidence._read_url", fake_read_url)

    bundle = collect_evidence(
        "Check this source https://example.com/report",
        enabled=True,
    )

    assert len(bundle.sources) == 1
    assert bundle.sources[0].verified is True
    assert bundle.sources[0].title == "Report"
    assert "Important source text" in bundle.to_prompt_block()


def test_collect_evidence_blocks_local_urls(monkeypatch):
    def fake_read_url(url, timeout):
        raise AssertionError("local URL should not be fetched")

    monkeypatch.setenv("COUNCIL_EVIDENCE_SEARCH", "0")
    monkeypatch.setattr("hermes_council.evidence._read_url", fake_read_url)

    bundle = collect_evidence("Check http://127.0.0.1/admin", enabled=True)

    assert bundle.sources == []
    assert "private or non-routable" in bundle.errors[0]


def test_collect_evidence_searches_and_fetches_result(monkeypatch):
    calls = []

    def fake_read_url(url, timeout):
        calls.append(url)
        if "duckduckgo.com/html" in url:
            return """
            <a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpaper">Paper</a>
            <div class="result__snippet">Search result snippet.</div>
            """
        return "<html><title>Paper</title><body>Fetched paper text with evidence.</body></html>"

    monkeypatch.setenv("COUNCIL_EVIDENCE_SEARCH", "1")
    monkeypatch.setattr("hermes_council.evidence._read_url", fake_read_url)

    bundle = collect_evidence("important research question", enabled=True, max_sources=1)

    assert len(bundle.sources) == 1
    assert bundle.sources[0].verified is True
    assert bundle.sources[0].url == "https://example.com/paper"
    assert any("duckduckgo.com/html" in call for call in calls)


def test_collect_evidence_keeps_unverified_search_result_when_fetch_fails(monkeypatch):
    def fake_read_url(url, timeout):
        if "duckduckgo.com/html" in url:
            return """
            <a class="result__a" href="https://example.com/down">Down</a>
            <div class="result__snippet">Fallback snippet.</div>
            """
        raise OSError("down")

    monkeypatch.setattr("hermes_council.evidence._read_url", fake_read_url)

    bundle = collect_evidence("question", enabled=True, max_sources=1)

    assert len(bundle.sources) == 1
    assert bundle.sources[0].verified is False
    assert bundle.verified_sources == []
    assert "down" in bundle.errors[0]


def test_prompt_block_marks_verified_and_unverified_sources():
    bundle = EvidenceBundle(
        [
            EvidenceSource("Verified", "https://example.com/v", "Verified snippet", "provided_url", True),
            EvidenceSource("Search", "https://example.com/s", "Search snippet", "search_result", False),
        ],
        [],
    )

    prompt = bundle.to_prompt_block()
    assert "verified" in prompt
    assert "unverified-search-result" in prompt
