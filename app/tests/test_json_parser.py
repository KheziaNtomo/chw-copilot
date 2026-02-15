"""Tests for JSON response parser.

Tests parse_json_response() from models.py which handles:
- Direct JSON strings
- JSON in markdown code fences
- JSON embedded in text  
- Unparseable inputs → None
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import parse_json_response


class TestParseJsonResponse:
    """Tests for robust JSON extraction from model outputs."""

    def test_direct_json(self):
        """Clean JSON string should parse directly."""
        text = '{"syndrome_tag": "respiratory_fever", "confidence": "high"}'
        result = parse_json_response(text)
        assert result is not None
        assert result["syndrome_tag"] == "respiratory_fever"

    def test_json_in_code_fence(self):
        """JSON wrapped in ```json code fence should be extracted."""
        text = 'Here is the result:\n```json\n{"syndrome_tag": "acute_watery_diarrhea"}\n```\n'
        result = parse_json_response(text)
        assert result is not None
        assert result["syndrome_tag"] == "acute_watery_diarrhea"

    def test_json_in_generic_fence(self):
        """JSON in ``` code fence (no language tag) should be extracted."""
        text = 'Result:\n```\n{"value": "yes", "evidence_quote": "fever"}\n```'
        result = parse_json_response(text)
        assert result is not None
        assert result["value"] == "yes"

    def test_json_embedded_in_text(self):
        """JSON object embedded in surrounding text should be found."""
        text = 'I found the following: {"syndrome_tag": "other"} based on the note.'
        result = parse_json_response(text)
        assert result is not None
        assert result["syndrome_tag"] == "other"

    def test_unparseable_returns_none(self):
        """Non-JSON text should return None."""
        text = "This is not JSON at all, just a plain text response."
        result = parse_json_response(text)
        assert result is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        result = parse_json_response("")
        assert result is None

    def test_nested_json(self):
        """Nested JSON objects should parse correctly."""
        text = '{"patient": {"age_years": 3, "sex": "male"}, "referral": true}'
        result = parse_json_response(text)
        assert result is not None
        assert result["patient"]["age_years"] == 3
        assert result["referral"] is True

    def test_json_with_preamble(self):
        """Model output with text before JSON should still extract it."""
        text = "Based on my analysis of the note, here is the structured output:\n\n{\"syndrome_tag\": \"respiratory_fever\", \"confidence\": \"high\", \"reasoning\": \"Fever + cough present\"}"
        result = parse_json_response(text)
        assert result is not None
        assert result["confidence"] == "high"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
