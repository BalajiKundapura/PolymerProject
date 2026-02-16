import re

from .normalization import normalize_for_parsing


def value_in_text(value: str, text: str) -> bool:
    if not value or not text:
        return False
    v = normalize_for_parsing(str(value)).lower()
    t = normalize_for_parsing(text).lower()
    if v in t:
        return True
    v_alt = v.replace("/", ":")
    if v_alt in t:
        return True
    v_alt2 = v.replace(":", "/")
    if v_alt2 in t:
        return True
    t_compact = re.sub(r"\s+", "", t)
    v_compact = re.sub(r"\s+", "", v)
    if v_compact in t_compact:
        return True
    return False
