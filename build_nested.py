# build_nested.py
import csv, json
from pathlib import Path

csv_path = Path(__file__).parent / "disease_translated.csv"
out_path = Path(__file__).parent / "nested_disease.json"

nested = {}
with open(csv_path, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        nested[row["key"]] = {
            "en": row["en_html"],
            "hi": row["hi_html"]
        }

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nested, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(nested)} entries to {out_path}")
