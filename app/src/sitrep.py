"""SITREP (Situation Report) generation using MedGemma or template fallback.

Generates structured weekly situation reports from anomaly detection results.
"""
import json
from typing import Dict, Any, List

import pandas as pd

from . import config


def _load_prompt() -> str:
    """Load the monitoring SITREP prompt template."""
    path = config.PROMPT_DIR / "monitoring_sitrep.txt"
    return path.read_text(encoding="utf-8")


def generate_sitrep_medgemma(
    anomalies: pd.DataFrame,
    weekly_counts: pd.DataFrame,
    week_id: int,
    locations: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Generate a weekly SITREP using MedGemma.

    Args:
        anomalies: DataFrame from detect_anomalies() — alerts for this week
        weekly_counts: Full weekly counts DataFrame
        week_id: The week to report on
        locations: Optional locations DataFrame with names

    Returns:
        SITREP dict matching sitrep.schema.json
    """
    from .models import generate_medgemma, parse_json_response

    prompt_template = _load_prompt()

    # Build anomaly summary for the prompt
    week_anomalies = anomalies[anomalies["week_id"] == week_id] if not anomalies.empty else anomalies
    week_counts = weekly_counts[weekly_counts["week_id"] == week_id] if not weekly_counts.empty else weekly_counts

    anomaly_text = "No anomalies detected." if week_anomalies.empty else ""
    for _, row in week_anomalies.iterrows():
        loc_name = row["location_id"]
        if locations is not None and "location_name" in locations.columns:
            match = locations[locations["location_id"] == row["location_id"]]
            if not match.empty:
                loc_name = match.iloc[0]["location_name"]
        anomaly_text += (
            f"- {row['syndrome_tag']} at {loc_name}: "
            f"count={int(row['count'])}, baseline_mean={row['baseline_mean']:.1f}, "
            f"ratio={row['count']/max(row['baseline_mean'],1):.1f}x baseline\n"
        )

    # Summary stats
    total_encounters = int(week_counts["count"].sum()) if not week_counts.empty else 0
    n_locations = week_counts["location_id"].nunique() if not week_counts.empty else 0

    prompt = prompt_template.replace("{week_id}", str(week_id))
    prompt = prompt.replace("{anomaly_data}", anomaly_text)
    prompt = prompt.replace("{total_encounters}", str(total_encounters))
    prompt = prompt.replace("{n_locations}", str(n_locations))

    raw_output = generate_medgemma(prompt, max_tokens=config.SITREP_MAX_TOKENS)
    parsed = parse_json_response(raw_output)

    if parsed is None:
        print(f"  WARNING: MedGemma SITREP returned unparseable output")
        return generate_sitrep_template(anomalies, weekly_counts, week_id, locations)

    # Ensure required fields
    parsed["week_id"] = week_id
    return parsed


def generate_sitrep_template(
    anomalies: pd.DataFrame,
    weekly_counts: pd.DataFrame,
    week_id: int,
    locations: pd.DataFrame = None,
) -> Dict[str, Any]:
    """Template-based SITREP generator (no model needed). Used as fallback.

    Produces a structured SITREP from the anomaly data.
    """
    week_anomalies = anomalies[anomalies["week_id"] == week_id] if not anomalies.empty else anomalies
    week_counts = weekly_counts[weekly_counts["week_id"] == week_id] if not weekly_counts.empty else weekly_counts

    total_encounters = int(week_counts["count"].sum()) if not week_counts.empty else 0
    n_locations = week_counts["location_id"].nunique() if not week_counts.empty else 0
    n_alerts = len(week_anomalies)

    # Build alerts
    alerts = []
    for _, row in week_anomalies.iterrows():
        loc_name = row["location_id"]
        if locations is not None and "location_name" in locations.columns:
            match = locations[locations["location_id"] == row["location_id"]]
            if not match.empty:
                loc_name = match.iloc[0]["location_name"]

        ratio = row["count"] / max(row["baseline_mean"], 1)
        alerts.append({
            "location": loc_name,
            "syndrome": row["syndrome_tag"],
            "count": int(row["count"]),
            "baseline_mean": round(row["baseline_mean"], 1),
            "ratio_to_baseline": round(ratio, 1),
            "action": "Investigate and verify counts; consider enhanced surveillance",
        })

    # Generate narrative
    if n_alerts == 0:
        narrative = (
            f"Week {week_id}: {total_encounters} syndromic encounters across "
            f"{n_locations} locations. No anomalies detected. "
            f"All syndrome counts within expected baseline ranges."
        )
    else:
        alert_summaries = []
        for a in alerts:
            alert_summaries.append(
                f"{a['syndrome'].replace('_', ' ')} at {a['location']} "
                f"({a['count']} cases, {a['ratio_to_baseline']}x baseline)"
            )
        narrative = (
            f"Week {week_id}: {total_encounters} syndromic encounters across "
            f"{n_locations} locations. {n_alerts} anomalies flagged: "
            + "; ".join(alert_summaries) + ". "
            f"Recommend field verification of elevated counts."
        )

    return {
        "week_id": week_id,
        "total_encounters": total_encounters,
        "reporting_locations": n_locations,
        "alerts": alerts,
        "narrative": narrative,
        "data_quality": {
            "completeness": "unknown",
            "notes": "Automated template — review recommended",
        },
        "limitations": [
            "Syndromic counts only — not confirmed diagnoses",
            "Anomaly threshold is deterministic (count >= baseline + 3, count >= 5)",
            "Small counts suppressed for privacy",
        ],
    }
