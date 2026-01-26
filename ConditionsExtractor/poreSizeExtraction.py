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
        select_main_polymers
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


nlp = spacy.load("en_core_web_sm")

PORE_SIZE_REGEX = re.compile(
    r"""
    \b
    (?:
        \d+(?:\.\d+)?        
        |
        \d+\s*(?:\^|\s)?\s*\d+       
    )
    \s*
    (?:Â\s*)?                     
    Å
    \b
    """,
    re.I | re.VERBOSE,
)

def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+", text)

def extract_pore_sizes(sentence):
    return [m.group().replace("Â", "").strip() for m in PORE_SIZE_REGEX.finditer(sentence)]

def is_pore_candidate(sentence):
    return PORE_SIZE_REGEX.search(sentence)

def extract_pore_conditions(sentences, window=2):
    records = []
    seen = set()

    for i, sent in enumerate(sentences):
        if not is_pore_candidate(sent):
            continue

        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end])

        all_polymers, _ = extract_all_polymers(context)
        main_polymers = select_main_polymers(all_polymers, {})

        if not main_polymers:
            continue

        pore_sizes = extract_pore_sizes(context)
        if not pore_sizes:
            continue

        key = (tuple(sorted(pore_sizes)), tuple(sorted(main_polymers.keys())))
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "pore_sizes": pore_sizes,
            "polymers": sorted(main_polymers.keys()),
            "context": context
        })

    return records

def write_output(records, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            f.write(f"===== EXTRACT {i} =====\n")
            f.write(f"Pore sizes: {r['pore_sizes']}\n")
            f.write(f"Polymers: {r['polymers']}\n")
            f.write("Context:\n")
            f.write(r["context"] + "\n\n")
    print(f"Extracted {len(records)} pore-size conditions")
    print(f"Saved to {output_file}")

def run_pipeline(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = split_sentences(text)
    records = extract_pore_conditions(sentences)
    write_output(records, output_file)

if __name__ == "__main__":
    run_pipeline(
        input_file="LCCC_pore_size.txt",
        output_file="LCCC_pore_size_conditions.txt"
    )
