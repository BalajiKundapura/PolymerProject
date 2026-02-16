import os
import re
from typing import List


def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input text not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_paragraphs(raw: str) -> List[str]:
    text = re.sub(r"\[[0-9]{1,3}\]", " ", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Split on blank lines OR on likely sentence boundaries created by PDF-to-text line wrapping.
    paras = re.split(r"\n\s*\n|(?<=[.!?])\s*\n+(?=[A-Z])", text)
    return [p.strip() for p in paras if p and len(p.strip()) > 20]


# Backwards compatibility alias.
clean_text = split_paragraphs
