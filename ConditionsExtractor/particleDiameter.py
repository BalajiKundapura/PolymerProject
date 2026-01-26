import re
import importlib.util
import sys
from pathlib import Path


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


def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+", text)

def extract_particle_values(sentence):
    return ["{} {}".format(m[0].strip(), m[1].strip()) for m in PARTICLE_REGEX.findall(sentence)]

def is_particle_candidate(sentence):
    return PARTICLE_REGEX.search(sentence)

def extract_particle_conditions(sentences, window=2):
    records = []
    seen = set()

    for i, sent in enumerate(sentences):
        if not is_particle_candidate(sent):
            continue

        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end])

        all_polymers, _ = extract_all_polymers(context)
        main_polymers = select_main_polymers(all_polymers, {})

        if not main_polymers:
            continue

        particle_values = extract_particle_values(context)
        if not particle_values:
            continue

        key = (tuple(sorted(particle_values)), tuple(sorted(main_polymers.keys())))
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "values": particle_values,
            "polymers": sorted(main_polymers.keys()),
            "context": context
        })

    return records


def write_output(records, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            f.write(f"===== EXTRACT {i} =====\n")
            f.write(f"Values: {r['values']}\n")
            f.write(f"Polymers: {r['polymers']}\n")
            f.write("Context:\n")
            f.write(r["context"] + "\n\n")
    print(f"Extracted {len(records)} particle-diameter conditions")
    print(f"Saved to {output_file}")


def run_pipeline(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    sentences = split_sentences(text)
    records = extract_particle_conditions(sentences)
    write_output(records, output_file)

if __name__ == "__main__":
    run_pipeline(
        input_file="LCCC_particle_diameter.txt",
        output_file="LCCC_particle_diameter_conditions.txt"
    )
