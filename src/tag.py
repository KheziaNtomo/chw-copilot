"""Syndrome tagging stub.

Allowed tags: respiratory_fever, acute_watery_diarrhea, other_unclear
"""
from typing import Dict


ALLOWED = ["respiratory_fever", "acute_watery_diarrhea", "other_unclear"]


def stub_tag(encounter: Dict, note_text: str) -> str:
	s = encounter.get("symptoms", {})
	fever = s.get("fever", {}).get("value")
	cough = s.get("cough", {}).get("value")
	watery = s.get("watery_diarrhea", {}).get("value")

	if watery == "yes":
		return "acute_watery_diarrhea"
	if fever == "yes" or cough == "yes":
		return "respiratory_fever"
	return "other_unclear"
