# dump_to_csv.py
import csv
from app import disease_dic

# writes out a two-column CSV: key, English HTML
with open("diseases.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["key", "en_html"])
    for key, html in disease_dic.items():
        writer.writerow([key, html])
print(f"Wrote {len(disease_dic)} rows to diseases.csv")
