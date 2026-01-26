import re
import sys
from pathlib import Path
from collections import Counter
import importlib.util


PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


try:
    from polymerSubject import extract_all_polymers, select_main_polymers
except Exception:
    spec_path = Path(PROJECT_ROOT) / "polymerSubject.py"
    if not spec_path.exists():
        raise
    spec = importlib.util.spec_from_file_location("polymerSubject", str(spec_path))
    polymerSubject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(polymerSubject)
    extract_all_polymers = polymerSubject.extract_all_polymers
    select_main_polymers = polymerSubject.select_main_polymers

INPUT_FILE = "LCCC_flow_rate.txt"
OUTPUT_FILE = "LCCC_flow_rate_conditions.txt"
FLOW_REGEX = re.compile(r"\b\d*\.?\d+\s*mL/min\b", re.I)

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return re.sub(r"\s+", " ", f.read()).strip()

def split_sentences(text: str):
    return re.split(r"(?<=[.!?])\s+", text)


def extract_main_flow_condition(sentences, window=1):
    flow_counts = Counter()
    polymer_counts = Counter()
    context_map = {}

    for i, sent in enumerate(sentences):
        matches = FLOW_REGEX.findall(sent)
        if not matches:
            continue

        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end]).strip()

        all_polymers, _ = extract_all_polymers(context)
        main_polys = select_main_polymers(all_polymers, {})

        for f in matches:
            flow_counts[f] += 1
            if f not in context_map:
                context_map[f] = []
            context_map[f].append((context, main_polys))
            
            for p in main_polys.keys():
                polymer_counts[p] += 1

    if not flow_counts:
        return None

    main_flow, _ = flow_counts.most_common(1)[0]

    contexts_for_flow = context_map[main_flow]
    if not contexts_for_flow:
        return None

    most_common_poly = polymer_counts.most_common(1)
    if most_common_poly:
        main_poly = most_common_poly[0][0]
    else:
        main_poly = "<UNKNOWN>"

    main_context = None
    for ctx, polys in contexts_for_flow:
        if main_poly in polys:
            main_context = ctx
            break
    if main_context is None:
        main_context = contexts_for_flow[0][0]

    return {
        "values": [main_flow],
        "polymers": [main_poly],
        "context": main_context
    }

def write_flow_condition(record, path):
    with open(path, "w", encoding="utf-8") as f:
        if record:
            f.write("===== EXTRACT 1 =====\n")
            f.write(f"Values: {record['values']}\n")
            f.write(f"Polymers: {record['polymers']}\n")
            f.write("Context:\n")
            f.write(record["context"] + "\n\n")
            print(f"flow rate condition saved: {record['values'][0]} with polymer {record['polymers'][0]}")
        else:
            f.write("No flow rate found\n")
            print("No flow rate found")

def run_pipeline():
    text = load_text(INPUT_FILE)
    sentences = split_sentences(text)
    main_record = extract_main_flow_condition(sentences)
    write_flow_condition(main_record, OUTPUT_FILE)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_pipeline()
