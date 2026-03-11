"""Tests for regex fallback parsers."""

from hermes_council.parsing import (
    parse_confidence,
    parse_dissent,
    parse_key_points,
    extract_sources,
    parse_persona_response,
)
from hermes_council.personas import PersonaResponse


class TestParseConfidence:
    def test_decimal(self):
        assert parse_confidence("CONFIDENCE: 0.85") == 0.85

    def test_percentage(self):
        assert parse_confidence("CONFIDENCE: 85%") == 0.85

    def test_integer_over_one(self):
        assert parse_confidence("Confidence: 72") == 0.72

    def test_missing(self):
        assert parse_confidence("No confidence here") == 0.5


class TestParseDissent:
    def test_true(self):
        assert parse_dissent("DISSENT: true") is True

    def test_false(self):
        assert parse_dissent("DISSENT: false") is False

    def test_case_insensitive(self):
        assert parse_dissent("Dissent: True") is True

    def test_missing(self):
        assert parse_dissent("No dissent field") is False


class TestParseKeyPoints:
    def test_bullet_points(self):
        text = "Header\n- First important point here\n- Second point here\n- Short"
        points = parse_key_points(text)
        assert len(points) == 2

    def test_asterisk_bullets(self):
        text = "* A significant finding\n* Another finding here"
        points = parse_key_points(text)
        assert len(points) == 2

    def test_max_ten(self):
        text = "\n".join(f"- Point number {i} with enough text" for i in range(15))
        points = parse_key_points(text)
        assert len(points) == 10


class TestExtractSources:
    def test_urls(self):
        text = "See https://example.com and http://test.org/page for details."
        sources = extract_sources(text)
        assert len(sources) == 2

    def test_deduplication(self):
        text = "https://example.com and again https://example.com"
        sources = extract_sources(text)
        assert len(sources) == 1

    def test_no_urls(self):
        assert extract_sources("No links here") == []


class TestParsePersonaResponse:
    def test_full_parse(self):
        text = (
            "Analysis here.\n"
            "- Key point one for analysis\n"
            "- Key point two for analysis\n"
            "CONFIDENCE: 0.8\n"
            "DISSENT: true\n"
            "Source: https://example.com"
        )
        resp = parse_persona_response("skeptic", text)
        assert isinstance(resp, PersonaResponse)
        assert resp.persona_name == "skeptic"
        assert resp.confidence == 0.8
        assert resp.dissents is True
        assert len(resp.key_points) == 2
        assert len(resp.sources) == 1
