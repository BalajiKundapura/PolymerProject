import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict
import spacy
import sys, os
import nltk


nltk.download('words')
from nltk.corpus import words
english_words = set(words.words())


nlp = spacy.load("en_core_web_sm")

if __package__ is None:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from polymerSubject import extract_all_polymers, validate_polymers, select_main_polymers


RATIO_RE = re.compile(r'(\d{1,3}\s*[:/]\s*\d{1,3}|\d{1,3}\s*vol%|\d{1,3}\s*%)')
SENTENCE_SPLIT_RE = re.compile(r'(?<!\bFig)(?<!\bEq)(?<!\bRef)(?<!\bNo)\.\s+')


def split_sentences(text: str) -> List[str]:
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sentences if s.strip()]


def extract_ratios(sent: str) -> List[str]:
    return RATIO_RE.findall(sent)


def extract_dynamic_chemicals(sentence: str, polymers: List[str]) -> List[str]:

    solvent_keywords = [
        "solvent", "eluent", "mobile phase", "elution",
        "dissolved in", "prepared in", "mixed with", "using",
        "blend", "injection", "mixture"
    ]

    sentence_lower = sentence.lower()
    polymer_lower = [p.lower() for p in polymers]

 
    if not any(kw in sentence_lower for kw in solvent_keywords):
        return []

    doc = nlp(sentence)
    chemicals = set()

    for token in doc:
        text = token.text.strip()
        text_lower = text.lower()

        if text_lower in polymer_lower:
            continue

        if not re.search(r'[a-z]', text):
            continue

        if not re.match(r'^[A-Za-z0-9/\-]+$', text):
            continue

        if text_lower in english_words:
            continue

        chemicals.add(text)

    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_lower = chunk_text.lower()
        if chunk_lower in polymer_lower or chunk_lower in english_words:
            continue
        if re.search(r'[a-z]', chunk_text) and re.match(r'^[A-Za-z0-9/\-]+$', chunk_text):
            chemicals.add(chunk_text)

    return list(chemicals)


def associate_solvents_with_polymers(sentences: List[str], polymers: List[str]):
    polymer_map = defaultdict(lambda: {"solvents": set(), "ratios": set(), "count": 0})
    polymer_set = set(polymers)

    for polymer in polymer_set:
        polymer_map[polymer]["count"] = 1
        for sent in sentences:
            if polymer.lower() in sent.lower():
                chemicals = extract_dynamic_chemicals(sent, polymers)
                if chemicals:
                    ratios = extract_ratios(sent)
                    polymer_map[polymer]["solvents"].update(chemicals)
                    polymer_map[polymer]["ratios"].update(ratios)

    for polymer in polymer_map:
        polymer_map[polymer]["solvents"] = list(polymer_map[polymer]["solvents"])
        polymer_map[polymer]["ratios"] = list(polymer_map[polymer]["ratios"])
    return dict(polymer_map)


def extract_polymers_solvents_dynamic(text: str) -> Dict:
    sentences = split_sentences(text)
    all_polymers, micro_dict = extract_all_polymers(text)
    validated_polymers = validate_polymers(all_polymers, micro_dict)
    main_polymers = select_main_polymers(validated_polymers, micro_dict)

    polymer_data = associate_solvents_with_polymers(sentences, main_polymers)

    for p in polymer_data:
        polymer_data[p]["count"] = validated_polymers.get(p, 0)

    return polymer_data

if __name__ == "__main__":
    sample_file = r"rawData\paper.txt"
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()

    extracted_data = extract_polymers_solvents_dynamic(text)

    for polymer, info in extracted_data.items():
        print(f"{polymer}:")
        print(f"  Count: {info['count']}")
        print(f"  Solvents: {info['solvents']}")
        print(f"  Ratios: {info['ratios']}")
        print()
