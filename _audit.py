"""Pre-upload audit: check notebook + support files for issues."""
import json, os, sys

ROOT = r"c:\Users\khezia\Documents\medGemma"
issues = []

# 1. Check notebook exists and has cells
nb_path = os.path.join(ROOT, "kaggle-main-output.ipynb")
nb = json.load(open(nb_path, encoding="utf-8"))
print(f"Notebook: {len(nb['cells'])} cells")

# 2. Check for stale model references
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    # Check for old model name
    if "medgemma-4b-it" in src.lower() and "1.5" not in src:
        issues.append(f"Cell {i}: references medgemma-4b-it (not 1.5)")
    # Check for old max_new_tokens=512 
    if "max_new_tokens=512" in src or "max_new_tokens = 512" in src:
        issues.append(f"Cell {i}: still uses max_new_tokens=512 (should be 1024)")
    # Check for offline-first language
    if "offline-first" in src.lower() or "offline first" in src.lower():
        issues.append(f"Cell {i}: contains 'offline-first' language")

# 3. Check if notebook loads pipeline_helpers from dataset or local
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "pipeline_helpers" in src:
        print(f"  Cell {i}: references pipeline_helpers")
        if "/kaggle/input" in src:
            print(f"    -> Loads from dataset (make sure dataset is updated!)")

# 4. Check prompt files
prompt_dir = os.path.join(ROOT, "prompts")
for fname in os.listdir(prompt_dir):
    fpath = os.path.join(prompt_dir, fname)
    content = open(fpath, encoding="utf-8").read()
    print(f"Prompt: {fname} ({len(content)} chars)")
    if "{note_text}" not in content and "combined" in fname:
        issues.append(f"Prompt {fname}: missing {{note_text}} placeholder")

# 5. Check pipeline_helpers.py for syntax errors
ph_path = os.path.join(ROOT, "src", "pipeline_helpers.py")
try:
    with open(ph_path, encoding="utf-8") as f:
        compile(f.read(), ph_path, "exec")
    print(f"pipeline_helpers.py: syntax OK ({os.path.getsize(ph_path)} bytes)")
except SyntaxError as e:
    issues.append(f"pipeline_helpers.py: SYNTAX ERROR at line {e.lineno}: {e.msg}")

# 6. Check support_files.zip exists
zip_path = os.path.join(ROOT, "support_files.zip")
if os.path.exists(zip_path):
    print(f"support_files.zip: {os.path.getsize(zip_path)} bytes")
else:
    issues.append("support_files.zip: MISSING")

# 7. Check gold_encounters exists
for subdir in ["data_synth"]:
    ge_path = os.path.join(ROOT, subdir, "gold_encounters.jsonl")
    if os.path.exists(ge_path):
        lines = open(ge_path, encoding="utf-8").readlines()
        print(f"{subdir}/gold_encounters.jsonl: {len(lines)} encounters")
    ge_merged = os.path.join(ROOT, subdir, "gold_encounters_merged.jsonl")
    if os.path.exists(ge_merged):
        lines = open(ge_merged, encoding="utf-8").readlines()
        print(f"{subdir}/gold_encounters_merged.jsonl: {len(lines)} encounters")

# 8. Check schemas
schema_dir = os.path.join(ROOT, "schemas")
if os.path.exists(schema_dir):
    for fname in os.listdir(schema_dir):
        fpath = os.path.join(schema_dir, fname)
        if fname.endswith(".json"):
            try:
                json.load(open(fpath, encoding="utf-8"))
                print(f"Schema: {fname} (valid JSON)")
            except json.JSONDecodeError as e:
                issues.append(f"Schema {fname}: INVALID JSON - {e}")

# Summary
print(f"\n{'='*50}")
if issues:
    print(f"ISSUES FOUND ({len(issues)}):")
    for iss in issues:
        print(f"  !! {iss}")
else:
    print("NO ISSUES FOUND - ready for upload")
