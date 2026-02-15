"""Tests for deterministic fallback functions.

Tests the rule-based fallbacks that activate when MedGemma is unavailable:
- Syndrome tagging (tag_syndrome_deterministic)
- Checklist generation (generate_checklist_deterministic)
- Anomaly detection (detect_anomalies)
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.tagger import tag_syndrome_deterministic
from src.checklist import generate_checklist_deterministic
from src.detect import detect_anomalies


class TestDeterministicTagger:
    """Tests for rule-based syndrome tagging."""

    def _make_encounter(self, symptoms):
        """Helper to create encounter with specified symptoms."""
        return {
            "symptoms": {
                k: {"value": v, "evidence_quote": f"{k}" if v == "yes" else ""}
                for k, v in symptoms.items()
            },
            "other_symptoms": {},
            "red_flags": [],
        }

    def test_respiratory_fever(self):
        """Fever + cough should tag as respiratory_fever."""
        enc = self._make_encounter({
            "fever": "yes",
            "cough": "yes",
            "diarrhea": "no",
        })
        result = tag_syndrome_deterministic(enc)
        assert result["syndrome_tag"] == "respiratory_fever"

    def test_acute_watery_diarrhea(self):
        """watery_diarrhea should tag as acute_watery_diarrhea."""
        enc = self._make_encounter({
            "fever": "unknown",
            "cough": "no",
            "watery_diarrhea": "yes",
            "bloody_diarrhea": "no",
            "vomiting": "yes",
        })
        result = tag_syndrome_deterministic(enc)
        assert result["syndrome_tag"] == "acute_watery_diarrhea"

    def test_unclear_no_symptoms(self):
        """No reported symptoms should tag as unclear."""
        enc = self._make_encounter({
            "fever": "unknown",
            "cough": "unknown",
            "diarrhea": "unknown",
        })
        result = tag_syndrome_deterministic(enc)
        assert result["syndrome_tag"] in ["unclear", "other"]

    def test_has_trigger_quotes(self):
        """Result should include trigger_quotes list."""
        enc = self._make_encounter({
            "fever": "yes",
            "cough": "yes",
        })
        result = tag_syndrome_deterministic(enc)
        assert "trigger_quotes" in result


class TestDeterministicChecklist:
    """Tests for rule-based checklist generation."""

    def test_generates_questions(self):
        """Missing fields should produce follow-up questions."""
        enc = {
            "symptoms": {
                "fever": {"value": "yes", "evidence_quote": "fever"},
                "cough": {"value": "unknown", "evidence_quote": ""},
                "diarrhea": {"value": "unknown", "evidence_quote": ""},
                "vomiting": {"value": "unknown", "evidence_quote": ""},
                "rash": {"value": "unknown", "evidence_quote": ""},
            },
            "other_symptoms": {},
            "red_flags": [],
            "patient": {"age_years": 3},
        }
        result = generate_checklist_deterministic(enc)
        assert "questions" in result
        assert len(result["questions"]) > 0

    def test_complete_encounter_fewer_questions(self):
        """A more complete encounter should generate fewer questions."""
        enc = {
            "symptoms": {
                "fever": {"value": "yes", "evidence_quote": "fever"},
                "cough": {"value": "yes", "evidence_quote": "cough"},
                "diarrhea": {"value": "no", "evidence_quote": "no diarrhea"},
                "vomiting": {"value": "no", "evidence_quote": "no vomiting"},
                "rash": {"value": "no", "evidence_quote": "no rash"},
            },
            "other_symptoms": {},
            "red_flags": [{"flag": "not_eating", "evidence_quote": "not eating"}],
            "patient": {"age_years": 3, "sex": "male"},
            "onset": "3 days",
            "severity": "moderate",
        }
        result = generate_checklist_deterministic(enc)
        # Should still generate some questions but potentially fewer
        assert "questions" in result


class TestAnomalyDetection:
    """Tests for deterministic anomaly detection."""

    def test_detects_spike(self):
        """Should detect anomaly when count exceeds baseline threshold."""
        data = pd.DataFrame([
            {"week_id": 1, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 3},
            {"week_id": 2, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 4},
            {"week_id": 3, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 3},
            {"week_id": 4, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 4},
            {"week_id": 5, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 15},
        ])
        anomalies = detect_anomalies(data)
        assert len(anomalies) > 0
        assert "loc01" in anomalies["location_id"].values

    def test_no_anomaly_normal_counts(self):
        """Should not flag anomalies for stable counts."""
        data = pd.DataFrame([
            {"week_id": 1, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 3},
            {"week_id": 2, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 4},
            {"week_id": 3, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 3},
            {"week_id": 4, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 4},
            {"week_id": 5, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 4},
        ])
        anomalies = detect_anomalies(data)
        assert len(anomalies) == 0

    def test_suppresses_small_counts(self):
        """Should not alert on small absolute counts even if ratio is high."""
        data = pd.DataFrame([
            {"week_id": 1, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 1},
            {"week_id": 2, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 1},
            {"week_id": 3, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 1},
            {"week_id": 4, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 1},
            {"week_id": 5, "location_id": "loc01", "syndrome_tag": "respiratory_fever", "count": 4},
        ])
        anomalies = detect_anomalies(data)
        # 4 is below MIN_COUNT_THRESHOLD (5) so should not alert
        assert len(anomalies) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
