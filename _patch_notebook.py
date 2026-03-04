"""Remove remaining bitsandbytes from pip install cell."""
import json

nb_path = r"c:\Users\khezia\Documents\medGemma\notebooks\kaggle_main.ipynb"
nb = json.load(open(nb_path, encoding="utf-8"))

for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "bitsandbytes" in src.lower() and "pip" in src:
        # Remove the bitsandbytes line entirely
        lines = src.split("\n")
        new_lines = [l for l in lines if "bitsandbytes" not in l]
        # Also fix comment 
        new_lines = [l.replace("(bitsandbytes enables 4-bit quantisation)", "") if "bitsandbytes enables" in l else l for l in new_lines]
        new_src = "\n".join(new_lines)
        if new_src != src:
            c["source"] = [new_src]
            print(f"Cell {i}: removed bitsandbytes from pip install")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Done")
