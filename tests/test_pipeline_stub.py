"""End-to-end pipeline tests using the stub (no model) path.

Tests the full agentic pipeline with deterministic fallbacks:
- Single encounter processing
- Batch processing
- Surveillance pipeline (aggregation, anomaly detection, SITREP)
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import process_encounter, process_batch, run_surveillance


class TestSingleEncounter:
    """Tests for single encounter processing through the pipeline."""

    SAMPLE_NOTE = (
        "Child 3yo M fever 3 days cough bad rash on chest "
        "no diarrhea mother says not eating gave ORS referred health center"
    )

    def test_encounter_extraction(self):
        """Pipeline should extract encounter with correct metadata."""
        result = process_encounter(
            self.SAMPLE_NOTE,
            encounter_id="test_001",
            location_id="loc01",
            week_id=5,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        assert result["encounter"]["encounter_id"] == "test_001"
        assert result["encounter"]["location_id"] == "loc01"
        assert result["encounter"]["week_id"] == 5

    def test_symptom_extraction(self):
        """Pipeline should extract symptoms as structured dict."""
        result = process_encounter(
            self.SAMPLE_NOTE,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        symptoms = result["encounter"]["symptoms"]
        assert isinstance(symptoms, dict)
        # Fever should be detected by keyword extractor
        assert symptoms["fever"]["value"] in ("yes", "unknown")

    def test_syndrome_tag_present(self):
        """Pipeline should produce a syndrome tag."""
        result = process_encounter(
            self.SAMPLE_NOTE,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        assert "syndrome_tag" in result
        assert result["syndrome_tag"]["syndrome_tag"] in [
            "respiratory_fever", "acute_watery_diarrhea", "other", "unclear"
        ]
        assert result["syndrome_tag"]["confidence"] in ["high", "medium", "low"]

    def test_checklist_generated(self):
        """Pipeline should produce a checklist with questions."""
        result = process_encounter(
            self.SAMPLE_NOTE,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        assert "checklist" in result
        assert "questions" in result["checklist"]
        assert isinstance(result["checklist"]["questions"], list)

    def test_validation_report(self):
        """Pipeline should produce a validation report."""
        result = process_encounter(
            self.SAMPLE_NOTE,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        validation = result["validation"]
        assert "schema_valid" in validation
        assert "overall_pass" in validation

    def test_agent_trace_complete(self):
        """Pipeline should produce trace for all agents executed."""
        result = process_encounter(
            self.SAMPLE_NOTE,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        trace = result["agent_trace"]
        assert isinstance(trace, list)
        assert len(trace) >= 4  # extract, evidence, tag, checklist, validate
        agent_ids = [step["agent"] for step in trace]
        assert "extract" in agent_ids
        assert "evidence_enforce" in agent_ids
        assert "tag" in agent_ids
        assert "validate" in agent_ids

    def test_processing_time_recorded(self):
        """Pipeline should record total processing time."""
        result = process_encounter(
            self.SAMPLE_NOTE,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        assert "processing_time_s" in result
        assert result["processing_time_s"] >= 0


class TestBatchProcessing:
    """Tests for batch encounter processing."""

    BATCH_NOTES = [
        {"note_text": "woman 25yo fever 2 days watery diarrhea vomiting dehydrated sunken eyes",
         "encounter_id": "batch_001", "location_id": "loc02", "week_id": 7},
        {"note_text": "boy 8yo cough 5 days difficulty breathing chest indrawing high fever",
         "encounter_id": "batch_002", "location_id": "loc04", "week_id": 7},
        {"note_text": "man 40yo headache back pain no fever no cough no diarrhea",
         "encounter_id": "batch_003", "location_id": "loc01", "week_id": 7},
    ]

    def test_batch_returns_all_results(self):
        """Batch should return one result per input note."""
        results = process_batch(
            self.BATCH_NOTES,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        assert len(results) == len(self.BATCH_NOTES)

    def test_batch_encounter_ids_preserved(self):
        """Batch should preserve encounter IDs from input."""
        results = process_batch(
            self.BATCH_NOTES,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        ids = [r["encounter"]["encounter_id"] for r in results]
        assert "batch_001" in ids
        assert "batch_002" in ids
        assert "batch_003" in ids

    def test_batch_each_has_syndrome_tag(self):
        """Every result in batch should have a syndrome tag."""
        results = process_batch(
            self.BATCH_NOTES,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )
        for r in results:
            assert "syndrome_tag" in r
            assert r["syndrome_tag"]["syndrome_tag"] in [
                "respiratory_fever", "acute_watery_diarrhea", "other", "unclear"
            ]


class TestSurveillancePipeline:
    """Tests for the surveillance pipeline (aggregation + anomaly detection + SITREP)."""

    def _get_batch_results(self):
        """Run a batch and return results for surveillance."""
        notes = [
            {"note_text": "child fever cough", "encounter_id": f"s_{i}",
             "location_id": "loc01", "week_id": 7}
            for i in range(3)
        ]
        return process_batch(
            notes,
            extractor="stub",
            use_model_tagger=False,
            use_model_checklist=False,
            run_hallucination_check=False,
        )

    def test_surveillance_returns_expected_keys(self):
        """Surveillance output should contain weekly_counts, anomalies, sitreps."""
        results = self._get_batch_results()
        surveillance = run_surveillance(results, use_model_sitrep=False)
        assert "weekly_counts" in surveillance
        assert "anomalies" in surveillance
        assert "sitreps" in surveillance

    def test_weekly_counts_dataframe(self):
        """Weekly counts should be a DataFrame with expected columns."""
        results = self._get_batch_results()
        surveillance = run_surveillance(results, use_model_sitrep=False)
        wc = surveillance["weekly_counts"]
        assert hasattr(wc, "columns")
        assert "week_id" in wc.columns
        assert "location_id" in wc.columns
        assert "syndrome_tag" in wc.columns
        assert "count" in wc.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
