"""Generate synthetic gold encounters and simulation events for the CHW Copilot.

Gold encounters: 60 realistic CHW-style notes inspired by AfriMed-QA clinical
presentations (malaria, pneumonia, cholera, measles, gastroenteritis, etc.)
rewritten as short, telegraphic field notes.

Simulation events: 750 events across 10 weeks and 8 locations, with an outbreak
of respiratory_fever injected at loc04+loc05 in weeks 7-8.

Usage:
    python data_synth/generate.py
"""
import csv
import json
import random
from pathlib import Path

random.seed(42)

OUT = Path(__file__).parent
LOCATIONS = ["loc01", "loc02", "loc03", "loc04", "loc05", "loc06", "loc07", "loc08"]
SYNDROMES = ["respiratory_fever", "acute_watery_diarrhea", "other", "unclear"]
WEEKS = list(range(1, 11))  # weeks 1-10

# ─────────────────────────────────────────────────────────────
# PART A: 60 gold encounters — realistic messy CHW field notes
#         inspired by sub-Saharan African clinical presentations
# ─────────────────────────────────────────────────────────────

GOLD_NOTES = [
    # --- respiratory_fever (25 notes) ---
    {
        "note": "5yo boy fever 3 days, cough, rapid breathing. mother says hot body at night. no rash. eating OK but tired.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["high_fever"],
        "age_group": "child", "sex": "male",
    },
    {
        "note": "woman 30yr cough 1 week, fever on and off, chest pain when breathing deep. no diarrhea no vomiting.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "infant 8mo, mother reports hot body since yesterday, refused breast milk this morning, fast breathing noticed. no rash no diarrhea.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["chest_indrawing"],
        "age_group": "infant", "sex": "unknown",
    },
    {
        "note": "pt male 45 fever and headache 2 days. dry cough. lives near swamp area. no vomiting.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "girl 3yr high fever 39C, cough with noisy breathing, chest pulling in. not eating well. siblings also sick.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["high_fever", "chest_indrawing"],
        "age_group": "child", "sex": "female",
    },
    {
        "note": "elderly man approx 65, coughing blood-streaked sputum, feverish, losing weight. referred to district hospital.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["high_fever"],
        "age_group": "elderly", "sex": "male",
    },
    {
        "note": "teenager 15yr, sore throat and fever 2 day, mild cough. eating and drinking OK. no danger signs.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adolescent", "sex": "unknown",
    },
    {
        "note": "child 2yr hot body, running nose, cough, difficulty breathing noticed by mother. no diarrhea. immunization not up to date.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["chest_indrawing"],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "female 28, fever chills body pain 4 days. tested RDT negative for malaria yesterday at pharmacy. still coughing.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "boy 7yr fever cough 5 days not improving, fast breathing, mother says cant sleep at night from coughing. no vomiting no rash.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["high_fever"],
        "age_group": "child", "sex": "male",
    },
    {
        "note": "man 50 chronic cough 3wk, fever evening time, night sweats, weight loss. wife had TB last year.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "woman 25 pregnant 7mo, fever and cough started yesterday, mild headache. no bleeding no watery discharge.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "4yr girl hot body dry cough no runny nose. mother gave paracetamol at home. still feverish today.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "child", "sex": "female",
    },
    {
        "note": "baby 11mo fever 2 days, cough, pulling in of chest when breathing. not breastfeeding well. restless.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["chest_indrawing"],
        "age_group": "infant", "sex": "unknown",
    },
    {
        "note": "adult male farmer, hot body, headache, muscle aches, mild cough. rainy season. no diarrhea.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "child 6yr fever runny nose sneezing. no difficulty breathing no chest indrawing. playful. eating OK.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "woman elderly 70, cough worse at night, feverish, body aches. has diabetes. no diarrhea.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "elderly", "sex": "female",
    },
    {
        "note": "10yo boy boarding school, several students sick with fever and cough. this boy fever 38.5 dry cough 3d.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "child", "sex": "male",
    },
    {
        "note": "male 35, fever off and on 1 wk, productive cough yellow sputum, no blood. chest tight.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "girl 1yr high fever convulsion at home this morning. mother brought child, now alert but hot. cough present.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["convulsions", "high_fever"],
        "age_group": "infant", "sex": "female",
    },
    {
        "note": "man 40, feverish, general weakness, cough started 2 days ago. no rash no diarrhea. family history of asthma.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "adolescent girl 14, hot body sore throat cough 1 day. ate normally. no vomiting no diarrhea.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "adolescent", "sex": "female",
    },
    {
        "note": "child 4yr brought by grandmother. feverish, coughing, not eating since yesterday. grandmother says breathing fast.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["chest_indrawing"],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "adult 38 male, high fever shaking chills rigors. headache body pain cough. malaria RDT pending.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": ["high_fever"],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "baby 6mo, mother says hot body 1 day, cough and running nose. breastfeeding OK. no convulsion.",
        "gold_tag": "respiratory_fever",
        "gold_red_flags": [],
        "age_group": "infant", "sex": "unknown",
    },
    # --- acute_watery_diarrhea (20 notes) ---
    {
        "note": "child 3yr watery diarrhea 5 times since morning, vomiting twice. mother says no blood in stool. not drinking well.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs", "unable_to_drink"],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "woman 22 loose watery stool since yesterday, 4 episodes. mild stomach cramps. no fever checked. drinking ORS at home.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "man 55 sudden onset watery diarrhea and vomiting today. rice-water stools. becoming weak. lives near river.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs", "persistent_vomiting"],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "baby 10mo watery stool 6 times, vomiting everything given. sunken eyes noticed. mother very worried.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs", "persistent_vomiting", "unable_to_drink"],
        "age_group": "infant", "sex": "unknown",
    },
    {
        "note": "adult female 30 diarrhea 3 days, watery. no blood. mild fever. eating less but drinking water.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "elderly man 68, watery stool many times today, vomiting, too weak to stand. skin pinch slow return.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs", "persistent_vomiting"],
        "age_group": "elderly", "sex": "male",
    },
    {
        "note": "child 5yr loose stool 3 times. no vomiting. playful and drinking. mother asking for ORS.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "girl 2yr diarrhea watery 8 episodes since last night. vomiting. eyes sunken, mouth dry, very sleepy.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs", "altered_consciousness", "persistent_vomiting"],
        "age_group": "child", "sex": "female",
    },
    {
        "note": "pt male 40, running stomach since morning, watery stool no blood, mild cramps. ate leftover food yesterday.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "woman 35 diarrhea 2 days, watery, + vomiting. no fever. neighbor also has diarrhea. water source shared borehole.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "infant 7mo stool very watery greenish 4 times. breastfed only. no vomiting. fontanelle slightly sunken.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs"],
        "age_group": "infant", "sex": "unknown",
    },
    {
        "note": "boy 8yr stomach pain and watery diarrhea after school. vomited once. drinking water. no fever.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "child", "sex": "male",
    },
    {
        "note": "male 25 sudden diarrhea rice-water type, cramping, vomiting. co-workers also affected. ate same food at canteen.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["persistent_vomiting"],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "child 18mo diarrhea 7x today, watery, refuses to drink. crying with no tears. restless.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs", "unable_to_drink"],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "adolescent 16yr female, loose stool watery 3 days, abdominal cramps, no blood. mild nausea. drinking fluids OK.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "adolescent", "sex": "female",
    },
    {
        "note": "woman 40, watery diarrhea started night before, 5 episodes. vomiting x2. thirsty, drinking ORS. no fever.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "elderly woman 75, watery stools all day. too weak to eat. family says she is confused. dry mouth, sunken eyes.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs", "altered_consciousness"],
        "age_group": "elderly", "sex": "female",
    },
    {
        "note": "child 4yr brought from school, diarrhea watery 4x, no blood, mild tummy ache. active and drinking.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "man 28 diarrhea profuse watery no blood. started after drinking river water. vomiting present. weak.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": ["dehydration_signs", "persistent_vomiting"],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "baby 14mo loose watery stool. mother says 3 episodes. still breastfeeding. no vomiting. mild fever.",
        "gold_tag": "acute_watery_diarrhea",
        "gold_red_flags": [],
        "age_group": "infant", "sex": "unknown",
    },
    # --- other (10 notes) ---
    {
        "note": "child 6yr rash all over body, eyes red, fever 4 days. measles in village. cough present.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "woman 32, skin wound on left leg infected. swollen and red with pus. no fever. cleaning wound daily.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "man 48 diabetic, numbness in feet getting worse, blurred vision sometimes. no fever no cough.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "pregnant woman 26, headache and swollen legs. BP high reading at last ANC visit. no fever no cough no diarrhea.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "boy 9yr fell from tree, arm swollen very painful. cannot move it. no open wound. no fever.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "child", "sex": "male",
    },
    {
        "note": "girl 12yr severe stomach pain lower right, vomiting once. no diarrhea. no fever. pain getting worse.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "child", "sex": "female",
    },
    {
        "note": "adult male 37 painful urination 3 days. yellowish discharge. no fever. sexually active.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "child 5yr itchy scalp, hair falling out patches. no fever no cough. several children at school affected.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "woman 45 lump in breast noticed 2 months ago. getting bigger. no pain. no fever. referred.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "elderly 72 male, joint pain both knees getting worse. using walking stick. no history of fever.",
        "gold_tag": "other",
        "gold_red_flags": [],
        "age_group": "elderly", "sex": "male",
    },
    # --- unclear (5 notes) ---
    {
        "note": "pt came in feeling generally unwell. tired. no specific complaints. eating less.",
        "gold_tag": "unclear",
        "gold_red_flags": [],
        "age_group": "unknown", "sex": "unknown",
    },
    {
        "note": "child 3yr mother says not well since 2 days. no fever measured. eats little. no cough no stool problems.",
        "gold_tag": "unclear",
        "gold_red_flags": [],
        "age_group": "child", "sex": "unknown",
    },
    {
        "note": "man approx 50, body not right, headache. feverish maybe but no thermometer. no cough. stool normal.",
        "gold_tag": "unclear",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "male",
    },
    {
        "note": "woman 23 dizziness and fatigue. ate today. no vomiting. no diarrhea. no cough. might be pregnant.",
        "gold_tag": "unclear",
        "gold_red_flags": [],
        "age_group": "adult", "sex": "female",
    },
    {
        "note": "adolescent boy body aches 1 day. no fever yet. no cough. bowel OK. could be overwork from farm.",
        "gold_tag": "unclear",
        "gold_red_flags": [],
        "age_group": "adolescent", "sex": "male",
    },
]

def generate_gold_encounters():
    """Write gold_encounters.jsonl — 60 labeled notes for evaluation."""
    records = []
    for i, item in enumerate(GOLD_NOTES, start=1):
        loc = random.choice(LOCATIONS)
        week = random.choice(WEEKS)
        records.append({
            "encounter_id": f"gold_{i:03d}",
            "location_id": loc,
            "week_id": week,
            "note_text": item["note"],
            "gold_syndrome_tag": item["gold_tag"],
            "gold_red_flags": item["gold_red_flags"],
            "gold_age_group": item["age_group"],
            "gold_sex": item["sex"],
        })
    path = OUT / "gold_encounters.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} gold encounters → {path}")
    return records


# ─────────────────────────────────────────────────────────────
# PART B: 750 simulation events with outbreak injection
# ─────────────────────────────────────────────────────────────

def generate_sim_events():
    """Generate ~750 events with an outbreak at loc04+loc05 weeks 7-8."""
    rows = []
    eid = 0

    # Baseline rates per location per week
    BASE_RATES = {
        "respiratory_fever": 3,       # ~3 cases/loc/week normally
        "acute_watery_diarrhea": 2,   # ~2 cases/loc/week normally
        "other": 1,
    }

    for week in WEEKS:
        for loc in LOCATIONS:
            for syndrome, base in BASE_RATES.items():
                # Add some random variation (Poisson-like)
                count = max(0, base + random.randint(-1, 2))

                # OUTBREAK INJECTION: loc04 and loc05, weeks 7-8, respiratory_fever
                if loc in ("loc04", "loc05") and week in (7, 8) and syndrome == "respiratory_fever":
                    count += random.randint(12, 18)  # spike of +12-18

                for _ in range(count):
                    eid += 1
                    rows.append({
                        "encounter_id": f"sim_{eid:04d}",
                        "week_id": week,
                        "location_id": loc,
                        "syndrome_tag": syndrome,
                        "severity": random.choice(["mild", "moderate", "severe"]),
                    })

    path = OUT / "sim_events.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["encounter_id", "week_id", "location_id", "syndrome_tag", "severity"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} simulation events → {path}")
    return rows


if __name__ == "__main__":
    generate_gold_encounters()
    generate_sim_events()
