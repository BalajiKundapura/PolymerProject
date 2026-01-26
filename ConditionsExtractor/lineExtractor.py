import re
import csv
from typing import List, Dict


INPUT_FILE = "rawData\paper.txt"
TEXT_OUTPUT = "LCCC_context_blocks.txt"
CSV_OUTPUT = "LCCC_structured.csv"
WINDOW = 2

SOLVENTS = [
    ]

SOLVENT_KEYWORDS = [
    "solvent", "mobile phase", "eluent",
    "mixture", "consisting of"
]

RATIO_REGEX = re.compile(r"\b\d+\s*[/\:]\s*\d+\b")
PERCENT_REGEX = re.compile(r"\b\d+\s*(vol\s*%|wt\s*%)\b", re.I)
SOLVENT_PAIR_REGEX = re.compile(r"\b[a-zA-Z]+\s*/\s*[a-zA-Z]+\b")

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sentences(text: str) -> List[str]:
    return re.split(r"(?<=[.!?])\s+", text)

def sentence_has_signal(sentence: str) -> bool:
    s = sentence.lower()
    return (
        any(k in s for k in SOLVENT_KEYWORDS) or
        any(sol in s for sol in SOLVENTS) or
        RATIO_REGEX.search(sentence) or
        PERCENT_REGEX.search(sentence) or
        SOLVENT_PAIR_REGEX.search(sentence)
    )

def extract_contexts(sentences: List[str]) -> List[Dict]:
    results = []
    seen = set()

    for i, sent in enumerate(sentences):
        if sentence_has_signal(sent):
            start = max(0, i - WINDOW)
            end = min(len(sentences), i + WINDOW + 1)
            context = " ".join(sentences[start:end]).strip()

            if context in seen:
                continue
            seen.add(context)

            results.append({
                "sentence_index": i,
                "context": context,
                "core_sentence": sent
            })

    return results

def parse_structured(context_blocks: List[Dict]) -> List[Dict]:
    structured = []

    for block in context_blocks:
        text = block["context"].lower()

        solvents_found = sorted(
            {s for s in SOLVENTS if s in text}
        )

        ratios_found = RATIO_REGEX.findall(block["context"])
        percents_found = PERCENT_REGEX.findall(block["context"])
        solvent_pairs = SOLVENT_PAIR_REGEX.findall(block["context"])

        structured.append({
            "solvents": ", ".join(solvents_found),
            "solvent_pairs": ", ".join(solvent_pairs),
            "ratios": ", ".join(ratios_found),
            "percents": ", ".join(percents_found),
            "context": block["context"]
        })

    return structured

def write_txt(blocks: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for i, b in enumerate(blocks, 1):
            f.write(f"===== LCCC EXTRACT {i} =====\n")
            f.write(b["context"] + "\n\n")

def write_csv(rows: List[Dict], path: str):
    fieldnames = ["solvents", "solvent_pairs", "ratios", "percents", "context"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def run_pipeline():
    text = load_text(INPUT_FILE)
    sentences = split_sentences(text)
    context_blocks = extract_contexts(sentences)
    structured = parse_structured(context_blocks)

    write_txt(context_blocks, TEXT_OUTPUT)
    write_csv(structured, CSV_OUTPUT)

    print(f"✅ Context blocks extracted: {len(context_blocks)}")
    print(f"📄 Text output: {TEXT_OUTPUT}")
    print(f"📊 CSV output: {CSV_OUTPUT}")

if __name__ == "__main__":
    run_pipeline()
