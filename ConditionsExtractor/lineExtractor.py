import os
import sys
import re
import csv
from typing import List
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from polymerSubject import extract_all_polymers, select_main_polymers
except Exception:
    import importlib.util
    spec_path = Path(PROJECT_ROOT) / "polymerSubject.py"
    if not spec_path.exists():
        raise
    spec = importlib.util.spec_from_file_location("polymerSubject", str(spec_path))
    polymerSubject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(polymerSubject)
    extract_all_polymers = polymerSubject.extract_all_polymers
    select_main_polymers = polymerSubject.select_main_polymers


INPUT_FILE = "rawData/paper.txt"
WINDOW = 2

OUTPUT_CONFIG = {
    "mobile": ("LCCC_mobile_phase.txt", "LCCC_mobile_phase.csv"),
    "base": ("LCCC_base_material.txt", "LCCC_base_material.csv"),
    "particle": ("LCCC_particle_diameter.txt", "LCCC_particle_diameter.csv"),
    "pore": ("LCCC_pore_size.txt", "LCCC_pore_size.csv"),
    "temperature": ("LCCC_temperature.txt", "LCCC_temperature.csv"),
    "flow": ("LCCC_flow_rate.txt", "LCCC_flow_rate.csv")
}


SOLVENT_KEYWORDS = [
    "solvent", "mobile phase", "eluent",
    "mixture", "consisting of"
]

RATIO_REGEX = re.compile(r"\b\d+\s*[/\:]\s*\d+\b")
PERCENT_REGEX = re.compile(r"\b\d+\s*(vol\s*%|wt\s*%)\b", re.I)
SOLVENT_PAIR_REGEX = re.compile(r"\b[a-zA-Z]+\s*/\s*[a-zA-Z]+\b")

BASE_MATERIAL_KEYWORDS = [
    "silica", "polymer", "polystyrene", "divinylbenzene", "ps-dvb", "carbon"
]


PARTICLE_REGEX = re.compile(
    r"""
    \b
    (?P<value>\d+(?:\.\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)
    \s*
    (?P<unit>µm|um|micron|microns|nm|nanometer|nanometers)
    \b
    """,
    re.I | re.VERBOSE
)

PORE_REGEX = re.compile(r"\b\d+(\.\d+)?\s*(Å|A|angstroms?)\b", re.I)
TEMP_REGEX = re.compile(r"\b\d+(\.\d+)?\s*(°c|c)\b", re.I)
FLOW_REGEX = re.compile(r"\b\d+(\.\d+)?\s*(mL/min|ml/min)\b", re.I)



def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return re.sub(r"\s+", " ", text).strip()

def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)


def extract_particle_diameter(sentences, window=1):
    records, seen = [], set()
    for i, sent in enumerate(sentences):
        matches = PARTICLE_REGEX.findall(sent)
        if not matches:
            continue

        values = ["{} {}".format(m[0].strip(), m[1].strip()) for m in matches]
        values = list(sorted(set(values)))

        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end]).strip()

        key = (tuple(values), context)
        if key in seen:
            continue
        seen.add(key)

        all_polymers, _ = extract_all_polymers(context)
        main_polymers = select_main_polymers(all_polymers, {})

        records.append({
            "values": values,
            "polymers": sorted(main_polymers.keys()) if main_polymers else ["<UNKNOWN>"],
            "context": context
        })
    return records

def extract_keywords(sentences, keywords, regexes=None):
    results, seen = [], set()
    for i, sent in enumerate(sentences):
        found = any(k in sent.lower() for k in keywords)
        found |= any(r.search(sent) for r in regexes) if regexes else False
        if not found:
            continue

        start = max(0, i - WINDOW)
        end = min(len(sentences), i + WINDOW + 1)
        context = " ".join(sentences[start:end]).strip()

        if context in seen:
            continue
        seen.add(context)
        results.append({"context": context})
    return results


def write_txt(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            f.write(f"===== EXTRACT {i} =====\n")
            if "values" in r:
                f.write(f"Values: {r['values']}\n")
                f.write(f"Polymers: {r['polymers']}\n")
            f.write("Context:\n")
            f.write(r["context"] + "\n\n")

def write_csv(records, path):
    if not records:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        keys = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in records:
            row = r.copy()
            if "values" in r:
                row["values"] = ", ".join(r["values"])
                row["polymers"] = ", ".join(r["polymers"])
            writer.writerow(row)

def extract_regex(sentences, regex, window=1):
    records, seen = [], set()
    for i, sent in enumerate(sentences):
        matches = regex.findall(sent)
        if not matches:
            continue

        values = []
        for m in matches:
            if isinstance(m, tuple):
                values.append("".join(m).strip())
            else:
                values.append(str(m).strip())
        values = list(sorted(set(values)))

        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end]).strip()

        key = (tuple(values), context)
        if key in seen:
            continue
        seen.add(key)

        all_polymers, _ = extract_all_polymers(context)
        main_polymers = select_main_polymers(all_polymers, {})

        records.append({
            "values": values,
            "polymers": sorted(main_polymers.keys()) if main_polymers else ["<UNKNOWN>"],
            "context": context
        })
    return records


def mobile_trigger(s):
    s_low = s.lower()
    return (
        any(k in s_low for k in SOLVENT_KEYWORDS) or
        RATIO_REGEX.search(s) or
        PERCENT_REGEX.search(s) or
        SOLVENT_PAIR_REGEX.search(s)
    )

def base_trigger(s):
    return any(m in s.lower() for m in BASE_MATERIAL_KEYWORDS)


def run_pipeline():
    text = load_text(INPUT_FILE)
    sentences = split_sentences(text)

    mobile = extract_keywords(sentences, SOLVENT_KEYWORDS, regexes=[RATIO_REGEX, PERCENT_REGEX, SOLVENT_PAIR_REGEX])
    txt_path, csv_path = OUTPUT_CONFIG["mobile"]
    write_txt(mobile, txt_path)
    write_csv([{
        "solvent_pairs": ", ".join(SOLVENT_PAIR_REGEX.findall(b["context"])),
        "ratios": ", ".join(RATIO_REGEX.findall(b["context"])),
        "percents": ", ".join(PERCENT_REGEX.findall(b["context"])),
        "context": b["context"]
    } for b in mobile], csv_path)

    base = extract_keywords(sentences, BASE_MATERIAL_KEYWORDS)
    txt_path, csv_path = OUTPUT_CONFIG["base"]
    write_txt(base, txt_path)
    write_csv([{"base_material": ", ".join([k for k in BASE_MATERIAL_KEYWORDS if k in b["context"].lower()]),
                "context": b["context"]} for b in base], csv_path)

    particle = extract_particle_diameter(sentences, window=1)
    txt_path, csv_path = OUTPUT_CONFIG["particle"]
    write_txt(particle, txt_path)
    write_csv([{"particle_diameter": r["values"], "polymers": r["polymers"], "context": r["context"]} for r in particle], csv_path)

    pore = extract_regex(sentences, PORE_REGEX, window=1)
    txt_path, csv_path = OUTPUT_CONFIG["pore"]
    write_txt(pore, txt_path)
    write_csv([{"pore_size": r["values"], "polymers": r["polymers"], "context": r["context"]} for r in pore], csv_path)

    temp = extract_regex(sentences, TEMP_REGEX, window=1)
    txt_path, csv_path = OUTPUT_CONFIG["temperature"]
    write_txt(temp, txt_path)
    write_csv([{"temperature": r["values"], "polymers": r["polymers"], "context": r["context"]} for r in temp], csv_path)

    flow = extract_regex(sentences, FLOW_REGEX, window=1)
    txt_path, csv_path = OUTPUT_CONFIG["flow"]
    write_txt(flow, txt_path)
    write_csv([{"flow_rate": r["values"], "polymers": r["polymers"], "context": r["context"]} for r in flow], csv_path)

    print("LCCC extraction complete")

if __name__ == "__main__":
    run_pipeline()
