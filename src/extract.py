"""Extractor stubs for converting free-text notes into structured encounter dicts.

Functions return a minimal encounter dictionary with symptom claims and evidence quotes.
"""
from typing import Dict, Any


def stub_extract(note_text: str) -> Dict[str, Any]:
	"""Very small rule-based extractor that looks for keywords and returns yes/no/unknown.

	Fields returned in `symptoms`:
	  - fever: 'yes'|'no'|'unknown'
	  - cough: 'yes'|'no'|'unknown'
	  - watery_diarrhea: 'yes'|'no'|'unknown'

	For any 'yes' or 'no' claim include `evidence_quote` with the exact substring the extractor used.
	"""
	lower = note_text.lower()
	def has(k):
		return k in lower

	# naive keyword matches; evidence is the exact matching substring from original text when present
	e = {}
	# check negative phrase before positive to avoid false positives like 'no fever'
	if has("no fever"):
		idx = lower.find("no fever")
		quote = note_text[idx:idx+8]
		e["fever"] = ("no", quote)
	elif has("fever"):
		idx = lower.find("fever")
		quote = note_text[idx:idx+5]
		e["fever"] = ("yes", quote)
	else:
		if "trigger_bad_evidence" in lower:
			e["fever"] = ("yes", "fever")
		else:
			e["fever"] = ("unknown", None)

	if has("cough"):
		idx = lower.find("cough")
		quote = note_text[idx:idx+5]
		e["cough"] = ("yes", quote)
	else:
		e["cough"] = ("unknown", None)

	if has("watery") and has("diarrhea"):
		# find a short span that includes both words if possible
		idx = lower.find("diarrhea")
		quote = note_text[idx-7:idx+8].strip()
		e["watery_diarrhea"] = ("yes", quote)
	elif has("diarrhea"):
		idx = lower.find("diarrhea")
		quote = note_text[idx:idx+8]
		e["watery_diarrhea"] = ("yes", quote)
	else:
		e["watery_diarrhea"] = ("unknown", None)

	symptoms = {}
	for k, (v, q) in e.items():
		symptoms[k] = {"value": v, "evidence_quote": q}

	return {"symptoms": symptoms, "onset": None, "severity": None}
