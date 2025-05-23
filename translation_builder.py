# translation_builder.py

import json
from googletrans import Translator
from app import disease_dic

def build_translations(flat_dict):
    translator = Translator()
    nested = {}
    for key, en_html in flat_dict.items():
        print(f"Translating {key}…")
        hi_html = translator.translate(en_html, dest='hi').text
        nested[key] = {
            'en': en_html,
            'hi': hi_html
        }
    return nested

if __name__ == "__main__":
    translations = build_translations(disease_dic)
    # write to JSON
    with open("nested_disease.json", "w", encoding="utf-8") as f:
        json.dump(translations, f, ensure_ascii=False, indent=2)
    print("Wrote nested_disease.json with", len(translations), "entries.")
