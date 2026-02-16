import re
from typing import Optional


_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def normalize_for_parsing(text: str) -> str:
    """
    Normalize common PDF/OCR artifacts so extraction is more robust.
    This function is intentionally lightweight (no external deps).
    """
    if not text:
        return ""

    t = text
    # Normalize common PDF artifact where "x" becomes stray U+00C2 between numbers.
    t = re.sub(r"(\d)\s*\u00c2\s*(\d)", r"\1x\2", t)

    # Common unit / glyph normalizations
    t = t.replace("\u2013", "-")  # en dash
    t = t.replace("\u2014", "-")  # em dash
    t = t.replace("\u2212", "-")  # minus sign
    t = t.replace("\u03bc", "u")  # greek mu used for micro
    t = t.replace("\u00b5", "u")  # micro sign
    t = t.replace("\u00c5", "A")  # Angstrom sign
    t = t.replace("\u212b", "A")  # Angstrom symbol
    t = t.replace("\u00d7", "x")  # multiplication sign
    t = t.replace("\u00b0", "")  # degree sign
    t = t.replace("\u00ba", "")  # masculine ordinal often used as degree

    # OCR artifact: "particle diameterZ5 mm" or "pore sizeZ500"
    t = re.sub(r"(?<=\w)Z(?=\d)", " ", t)

    # OCR artifact: injection loop units rendered as "ml" for small volumes
    def _fix_loop_units(match: re.Match[str]) -> str:
        return f"{match.group(1)} uL loop"

    t = re.sub(r"\b(\d+(?:\.\d+)?)\s*m\s*l\s*loop\b", _fix_loop_units, t, flags=re.I)
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s*ml\s*loop\b", _fix_loop_units, t, flags=re.I)

    # OCR artifact: particle diameters shown as "mm" when context implies micrometers
    def _fix_particle_size(match: re.Match[str]) -> str:
        prefix = match.group(1)
        value = match.group(2)
        try:
            val = float(value)
        except Exception:
            return match.group(0)
        if val <= 20:
            return f"{prefix}{value} um"
        return match.group(0)

    t = re.sub(
        r"(particle\s+(?:diameter|size)[^0-9]{0,10})(\d+(?:\.\d+)?)\s*mm\b",
        _fix_particle_size,
        t,
        flags=re.I,
    )

    # Normalize percent unit spellings: "vol %" -> "vol%"
    t = re.sub(r"\b(vol|wt)\s*%", r"\1%", t, flags=re.I)

    # Common OCR: acetone spelled without 'c'
    t = re.sub(r"\baetone\b", "acetone", t, flags=re.I)
    t = re.sub(r"\baceton\b", "acetone", t, flags=re.I)

    # Common PDF spacing: "C 18 columns" -> "C18 columns"
    t = re.sub(r"\bC\s+(\d{1,2})\b(?=\s*(?:columns?|col\.|phase)\b)", r"C\1", t, flags=re.I)

    # Normalize common "-1" exponent unit formatting
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
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return str(val)
    if not isinstance(val, str):
        return None
    if val == "":
        return None
    return val.strip()
