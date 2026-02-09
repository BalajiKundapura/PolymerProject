import re
from typing import Optional


_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def normalize_for_parsing(text: str) -> str:
    """
    Normalize common PDF/OCR artifacts so regex-based extraction is more robust.
    This function is intentionally lightweight (no external deps).
    """
    if not text:
        return ""

    t = text
    # Normalize common PDF artifact where "x" becomes stray U+00C2 between numbers.
    t = re.sub(r"(\d)\s*\u00c2\s*(\d)", r"\1x\2", t)

    # Common unit / glyph normalizations
    t = t.replace("\u03bc", "u")  # greek mu used for micro
    t = t.replace("\u00b5", "u")  # micro sign
    t = t.replace("\u00c5", "A")  # Angstrom sign
    t = t.replace("\u212b", "A")  # Angstrom symbol
    t = t.replace("\u00d7", "x")  # multiplication sign
    t = t.replace("\u00b0", "")  # degree sign
    t = t.replace("\u00ba", "")  # masculine ordinal often used as degree

    # Normalize percent unit spellings: "vol %" -> "vol%"
    t = re.sub(r"\b(vol|wt)\s*%", r"\1%", t, flags=re.I)

    # Common PDF spacing: "C 18 columns" -> "C18 columns" (avoid mis-parsing as 18 columns)
    t = re.sub(r"\bC\s+(\d{1,2})\b(?=\s*(?:columns?|col\.|phase)\b)", r"C\1", t, flags=re.I)

    # Normalize common "-1" exponent unit formatting: "mL min -1" -> "mL/min", "mg mL -1" -> "mg/mL"
    t = re.sub(r"\b(mL|ml|uL)\s*min\s*-\s*1\b", r"\1/min", t, flags=re.I)
    t = re.sub(r"\b(mL|ml|uL)\s*min\s*-1\b", r"\1/min", t, flags=re.I)
    t = re.sub(r"\b(mg|ug|g)\s*mL\s*-\s*1\b", r"\1/mL", t, flags=re.I)
    t = re.sub(r"\b(mg|ug|g)\s*mL\s*-1\b", r"\1/mL", t, flags=re.I)
    t = re.sub(r"\b(mg|ug|g)\s*L\s*-\s*1\b", r"\1/L", t, flags=re.I)
    t = re.sub(r"\b(mg|ug|g)\s*L\s*-1\b", r"\1/L", t, flags=re.I)

    # Common OCR: "250!4.6" or "250! 4.6" used as "250 x 4.6"
    t = re.sub(r"(\d)\s*!\s*(\d)", r"\1x\2", t)

    # Common OCR: "25.0 8C" used as "25.0 C"
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s*8\s*([CFK])\b", r"\1 \2", t)

    # Replace remaining non-ascii with spaces to keep word boundaries sane
    t = _NON_ASCII_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def normalize_value(val: Optional[str]) -> Optional[str]:
    if val is None or val == "":
        return None
    return val.strip()
