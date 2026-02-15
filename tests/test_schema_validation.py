"""Tests for JSON schema validation.

Tests validate_json_schema() from validate.py against encounter and syndrome schemas.
Test data matches the actual schema requirements.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.validate import validate_json_schema


class TestEncounterSchema:
    """Tests for encounter schema validation."""

    def _make_valid_encounter(self, **overrides):
        """Create a valid encounter matching the actual schema."""
        enc = {
            "encounter_id": "test_001",
            "location_id": "loc_01",
            "week_id": 5,
            "chw_id": "chw_01",
            "area_id": "area_01",
            "household_id": "hh_01",
            "note_text": "child fever 3 days cough bad",
            "patient": {
                "age_years": 3,
                "age_group": "child",
                "sex": "male",
            },
            "symptoms": {
                "fever": {"value": "yes", "evidence_quote": "fever 3 days"},
                "cough": {"value": "no", "evidence_quote": None},
                "watery_diarrhea": {"value": "unknown", "evidence_quote": None},
                "bloody_diarrhea": {"value": "unknown", "evidence_quote": None},
                "vomiting": {"value": "unknown", "evidence_quote": None},
                "rash": {"value": "unknown", "evidence_quote": None},
                "difficulty_breathing": {"value": "unknown", "evidence_quote": None},
            },
            "other_symptoms": {},
            "onset_days": 3,
            "severity": "moderate",
            "red_flags": [],
            "referral": None,
        }
        enc.update(overrides)
        return enc

    def test_valid_encounter(self):
        """A complete valid encounter should pass validation."""
        encounter = self._make_valid_encounter()
        is_valid, errors = validate_json_schema(encounter, "encounter")
        assert is_valid, f"Valid encounter failed: {errors}"

    def test_missing_required_field(self):
        """Encounter missing required fields should fail validation."""
        encounter = {
            "encounter_id": "test_002",
        }
        is_valid, errors = validate_json_schema(encounter, "encounter")
        assert not is_valid
        assert len(errors) > 0

    def test_invalid_sex_value(self):
        """Patient with invalid sex value should fail."""
        encounter = self._make_valid_encounter(
            patient={"age_years": 3, "age_group": "child", "sex": "invalid_value"}
        )
        is_valid, errors = validate_json_schema(encounter, "encounter")
        assert not is_valid


class TestSyndromeSchema:
    """Tests for syndrome output schema validation."""

    def test_valid_syndrome(self):
        """A valid syndrome output should pass."""
        syndrome = {
            "encounter_id": "test_001",
            "syndrome_tag": "respiratory_fever",
            "confidence": "high",
            "reasoning": "Fever + cough present",
            "trigger_quotes": ["fever 3 days", "cough bad"],
        }
        is_valid, errors = validate_json_schema(syndrome, "syndrome")
        assert is_valid, f"Valid syndrome failed: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
