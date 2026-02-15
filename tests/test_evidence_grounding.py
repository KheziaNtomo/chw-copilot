"""Tests for evidence grounding enforcement.

Tests enforce_evidence() and enforce_trigger_quotes() from validate.py.
Verifies that:
- Valid verbatim quotes pass through unchanged
- Invalid/missing quotes cause downgrades to 'unknown'
- Edge cases (empty note, empty quote, partial match) are handled
"""
import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validate import enforce_evidence, enforce_trigger_quotes


class TestEnforceEvidence:
    """Tests for symptom evidence grounding."""

    def _make_encounter(self, symptoms=None, other_symptoms=None, red_flags=None):
        """Helper to create a minimal encounter dict."""
        enc = {
            "symptoms": symptoms or {},
            "other_symptoms": other_symptoms or {},
            "red_flags": red_flags or [],
        }
        return enc

    def test_valid_quotes_pass(self):
        """Symptoms with verbatim quotes from the note should pass."""
        note = "child has fever 3 days cough bad no diarrhea"
        encounter = self._make_encounter(symptoms={
            "fever": {"value": "yes", "evidence_quote": "fever 3 days"},
            "cough": {"value": "yes", "evidence_quote": "cough bad"},
            "diarrhea": {"value": "no", "evidence_quote": "no diarrhea"},
        })

        result, downgrades = enforce_evidence(encounter, note)

        assert len(downgrades) == 0
        assert result["symptoms"]["fever"]["value"] == "yes"
        assert result["symptoms"]["cough"]["value"] == "yes"

    def test_invalid_quote_downgraded(self):
        """Symptoms with quotes not found in note should be downgraded."""
        note = "child has fever 3 days no cough"
        encounter = self._make_encounter(symptoms={
            "fever": {"value": "yes", "evidence_quote": "fever 3 days"},
            "rash": {"value": "yes", "evidence_quote": "rash on chest"},  # not in note
        })

        result, downgrades = enforce_evidence(encounter, note)

        assert len(downgrades) > 0
        assert result["symptoms"]["rash"]["value"] == "unknown"

    def test_missing_quote_downgraded(self):
        """'yes' symptoms with empty evidence quotes should be downgraded."""
        note = "child has fever 3 days cough"
        encounter = self._make_encounter(symptoms={
            "fever": {"value": "yes", "evidence_quote": ""},
        })

        result, downgrades = enforce_evidence(encounter, note)

        assert len(downgrades) > 0
        assert result["symptoms"]["fever"]["value"] == "unknown"

    def test_no_value_unaffected(self):
        """Symptoms with value 'no' should not be affected."""
        note = "child has no fever"
        encounter = self._make_encounter(symptoms={
            "fever": {"value": "no", "evidence_quote": "no fever"},
        })

        result, downgrades = enforce_evidence(encounter, note)

        assert len(downgrades) == 0
        assert result["symptoms"]["fever"]["value"] == "no"

    def test_unknown_value_unaffected(self):
        """Symptoms with value 'unknown' should not be affected."""
        note = "child has cough"
        encounter = self._make_encounter(symptoms={
            "vomiting": {"value": "unknown", "evidence_quote": ""},
        })

        result, downgrades = enforce_evidence(encounter, note)

        assert len(downgrades) == 0

    def test_case_insensitive_matching(self):
        """Quote matching should be case-insensitive."""
        note = "Child has FEVER 3 Days"
        encounter = self._make_encounter(symptoms={
            "fever": {"value": "yes", "evidence_quote": "fever 3 days"},
        })

        result, downgrades = enforce_evidence(encounter, note)

        assert len(downgrades) == 0
        assert result["symptoms"]["fever"]["value"] == "yes"

    def test_red_flags_enforced(self):
        """Red flag evidence quotes should also be checked."""
        note = "child has fever not eating"
        encounter = self._make_encounter(
            symptoms={},
            red_flags=[
                {"flag": "not_eating", "evidence_quote": "not eating"},
                {"flag": "convulsions", "evidence_quote": "seizure"},  # not in note
            ],
        )

        result, downgrades = enforce_evidence(encounter, note)
        # Should flag convulsions as invalid
        assert any("convulsion" in d.lower() or "seizure" in d.lower() for d in downgrades)


class TestEnforceTriggerQuotes:
    """Tests for syndrome trigger quote enforcement."""

    def test_valid_trigger_quotes(self):
        """Trigger quotes found in note should pass."""
        note = "child has fever 3 days cough bad"
        syndrome = {
            "syndrome_tag": "respiratory_fever",
            "trigger_quotes": ["fever 3 days", "cough bad"],
        }

        result, invalid = enforce_trigger_quotes(syndrome, note)

        assert len(invalid) == 0

    def test_invalid_trigger_quotes(self):
        """Trigger quotes not in note should be flagged."""
        note = "child has fever 3 days"
        syndrome = {
            "syndrome_tag": "respiratory_fever",
            "trigger_quotes": ["fever 3 days", "difficulty breathing"],  # not in note
        }

        result, invalid = enforce_trigger_quotes(syndrome, note)

        assert len(invalid) > 0
        assert "difficulty breathing" in invalid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
