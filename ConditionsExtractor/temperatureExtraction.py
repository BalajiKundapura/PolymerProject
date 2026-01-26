import re
import spacy
import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from polymerSubject import (
        extract_all_polymers,
        select_main_polymers,
        normalize_name,
        polymer_abbrev_dict,
    )
except Exception:
    spec_path = Path(PROJECT_ROOT) / "polymerSubject.py"
    if not spec_path.exists():
        raise
    spec = importlib.util.spec_from_file_location("polymerSubject", str(spec_path))
    polymerSubject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(polymerSubject)
    extract_all_polymers = polymerSubject.extract_all_polymers
    select_main_polymers = polymerSubject.select_main_polymers
    normalize_name = polymerSubject.normalize_name
    polymer_abbrev_dict = getattr(polymerSubject, "polymer_abbrev_dict", {})

nlp = spacy.load("en_core_web_sm")

TEMP_REGEX = re.compile(
    r"""
    \b
    (?:temperature\s*(?:of|=)?\s*)?
    (?:
        (?P<num1>\d+(?:\.\d+)?)\s*°\s*(?P<unit1>C|K)   
        |
        °\s*(?P<unit2>C|K)\s*(?P<num2>\d+(?:\.\d+)?) 
    )
    \b
    """,
    re.I | re.VERBOSE,
)


TEMP_CONTEXT_TERMS = [
    "temperature",
    "critical",
    "near-critical",
    "conditions",
    "elution",
    "retention",
]


def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+", text)

def is_candidate(sentence):
    s = sentence.lower()
    return TEMP_REGEX.search(sentence) or any(term in s for term in TEMP_CONTEXT_TERMS)

def extract_temperatures(sentence):
    temps = []

    for m in TEMP_REGEX.finditer(sentence):
        if m.group("num1"):
            value = m.group("num1")
            unit = m.group("unit1")
        else:
            value = m.group("num2")
            unit = m.group("unit2")

        temps.append(f"{value} °{unit.upper()}")

    return temps

def extract_conditions(sentences, window=2):
    """
    Extract temperature conditions with linked polymers and context.
    """
    records = []
    seen = set()

    for i, sent in enumerate(sentences):
        if not is_candidate(sent):
            continue

        temperatures = extract_temperatures(sent)
        if not temperatures:
            continue

        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end])

        all_polymers, _ = extract_all_polymers(context)
        main_polymers = select_main_polymers(all_polymers, {})

        key = (tuple(temperatures), tuple(sorted(main_polymers.keys())))
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "temperatures": temperatures,
            "polymers": sorted(main_polymers.keys()) if main_polymers else ["<UNKNOWN>"],
            "context": context,
        })

    return records


def run_pipeline(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)
    records = extract_conditions(sentences)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            f.write(f"===== EXTRACT {i} =====\n")
            f.write(f"Temperatures: {r['temperatures']}\n")
            f.write(f"Polymers: {r['polymers']}\n")
            f.write("Context:\n")
            f.write(r["context"] + "\n\n")

    print(f"Extracted {len(records)} temperature conditions")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    run_pipeline(
        input_file="LCCC_temperature.txt",
        output_file="LCCC_temperature_conditions.txt",
    )
