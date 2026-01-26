import re
import spacy

nlp = spacy.load("en_core_web_sm")

RATIO_REGEX = re.compile(r"\b\d+\s*[/\:]\s*\d+\b")
UNIT_REGEX = re.compile(r"\b(vol\s*%|wt\s*%)\b", re.I)

MIXTURE_TERMS = [
    "mixture",
    "composition",
    "consisting",
    "mobile phase",
    "eluent",
    "solvent"
]

def split_sentences(text):
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?])\s+", text)

def is_candidate(sentence):
    s = sentence.lower()
    return (
        RATIO_REGEX.search(sentence)
        or "/" in sentence
        or any(term in s for term in MIXTURE_TERMS)
    )

def extract_solvent_phrases(sentence):
    doc = nlp(sentence)
    solvents = set()

    for match in re.findall(
        r"\b[a-zA-Z][a-zA-Z\-]+/[a-zA-Z][a-zA-Z\-]+\b",
        sentence
    ):
        solvents.add(match.lower())

    for chunk in doc.noun_chunks:
        text = chunk.text.lower()
        if " and " in text or " with " in text:
            solvents.add(text)

    for token in doc:
        if token.lemma_ == "mixture":
            for child in token.children:
                if child.dep_ == "prep":
                    phrase = " ".join(t.text for t in child.subtree)
                    solvents.add(phrase.lower())

    return sorted(solvents)

def extract_ratios(sentence):
    ratios = [m.group() for m in RATIO_REGEX.finditer(sentence)]
    unit_match = UNIT_REGEX.search(sentence)
    unit = unit_match.group() if unit_match else None
    return ratios, unit

def extract_conditions(sentences, window=2):
    records = []
    seen = set()

    for i, sent in enumerate(sentences):
        if not is_candidate(sent):
            continue

        solvents = extract_solvent_phrases(sent)
        ratios, unit = extract_ratios(sent)

        if not ratios:
            continue

        start = max(0, i - window)
        end = min(len(sentences), i + window + 1)
        context = " ".join(sentences[start:end])

        key = (tuple(solvents), tuple(ratios), unit)
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "solvent_expressions": solvents,
            "ratios": ratios,
            "unit": unit,
            "context": context
        })

    return records

def run_pipeline(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = split_sentences(text)
    records = extract_conditions(sentences)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(records, 1):
            f.write(f"===== LCCC CONDITION {i} =====\n")
            f.write(f"Solvent expressions: {r['solvent_expressions']}\n")
            f.write(f"Ratios: {r['ratios']}\n")
            f.write(f"Unit: {r['unit']}\n")
            f.write("Context:\n")
            f.write(r["context"] + "\n\n")

    print(f"✅ Extracted {len(records)} solvent–ratio conditions")
    print(f"📄 Saved to {output_file}")


if __name__ == "__main__":
    run_pipeline(
        input_file="LCCC_context_blocks.txt",
        output_file="LCCC_conditions.txt"
    )
