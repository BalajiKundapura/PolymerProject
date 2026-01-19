import re
from typing import List
from pathlib import Path

def normalize_units(text: str) -> str:
    text = re.sub(r'\b(mL|ml)\s*min\s*[−-]?\s*1\b', 'mL/min', text)
    text = re.sub(r'\b(μL|uL)\s*min\s*[−-]?\s*1\b', 'μL/min', text)

    text = re.sub(r'\b(mg)\s*(mL)\s*[−-]?\s*1\b', 'mg/mL', text)

    text = re.sub(r'°\s*C', '°C', text)

    text = re.sub(r'vol\s*%', 'vol%', text)
    text = re.sub(r'\bv/v\b', 'vol%', text)

    return text


def is_noise_sentence(sentence: str) -> bool:
    return bool(re.match(
        r'^(Figure|Table|Scheme|dx\.doi|Macromolecules|ARTICLE)',
        sentence.strip(),
        re.IGNORECASE
    ))



def split_sentences(text: str) -> List[str]:

    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(
        r'(?<!\bFig)(?<!\bEq)(?<!\bRef)(?<!\bNo)\.\s+',
        text
    )

    clean = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not s.endswith('.'):
            s += '.'
        clean.append(s)

    return clean



def preprocess_text(raw_text: str) -> List[str]:
    text = normalize_units(raw_text)
    sentences = split_sentences(text)
    return [s for s in sentences if not is_noise_sentence(s)]


def preprocess_file(input_path: str, output_path: str | None = None) -> List[str]:
    raw_text = Path(input_path).read_text(encoding="utf-8", errors="ignore")
    sentences = preprocess_text(raw_text)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text("\n".join(sentences), encoding="utf-8")

    return sentences


if __name__ == "__main__":
    sample_file = r"rawData\paper.txt"

    sentences = preprocess_file(sample_file)

    print("Preprocessed sentences from sample file:")
    for s in sentences:
        print(s)
