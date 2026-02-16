import re
from typing import Any, Dict, List, Tuple

from .text.normalization import normalize_for_parsing


SP_FIELDS = [
    "column_name",
    "material",
    "modification",
    "pore_size",
    "particle_diameter",
    "column_dimensions",
    "number_of_columns",
    "manufacturer",
    "phase",
]
SOL_FIELDS = ["solvent", "ratio", "ratio_units"]
TECH_FIELDS = [
    "temperature",
    "flow_rate",
    "injected_polymer_concentration",
    "injected_polymer_solvent_solution",
    "injection_volume",
]
SEP_FIELDS = ["critical_block_behavior", "non_critical_block_behavior", "purpose"]


def _norm(val: Any) -> str:
    if val is None:
        return ""
    return normalize_for_parsing(str(val)).lower()


def _merge_group(target: Dict[str, Any], source: Dict[str, Any], fields: List[str]) -> None:
    for f in fields:
        if target.get(f) in (None, "", {}):
            if source.get(f) not in (None, "", {}):
                target[f] = source.get(f)


def _parse_ratio_range(value: Any) -> Tuple[float, float] | None:
    if value in (None, ""):
        return None
    text = normalize_for_parsing(str(value))
    m = re.match(r"^\s*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*$", text)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) is not None else lo
    return (min(lo, hi), max(lo, hi))


def _parse_value_range(value: Any) -> Tuple[float, float, str] | None:
    if value in (None, ""):
        return None
    text = normalize_for_parsing(str(value))
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if not nums:
        return None
    lo = float(nums[0])
    hi = float(nums[1]) if len(nums) > 1 else lo
    unit = re.sub(r"[0-9\.\-\s]+", "", text).strip()
    return (min(lo, hi), max(lo, hi), unit)


def _merge_numeric_range(a_val: Any, b_val: Any) -> str | None:
    a = _parse_value_range(a_val)
    b = _parse_value_range(b_val)
    if not a or not b:
        return None
    lo = min(a[0], b[0])
    hi = max(a[1], b[1])
    unit = a[2] or b[2]
    if a[2] and b[2] and a[2] != b[2]:
        unit = a[2]
    unit_str = f" {unit}".strip()
    if lo == hi:
        return f"{lo:g} {unit}".strip()
    return f"{lo:g}-{hi:g} {unit}".strip()


def _format_solvent_name(name: str) -> str:
    if not name:
        return ""
    n = normalize_for_parsing(name).strip()
    lower = n.lower()
    if lower in {"thf", "tetrahydrofuran"}:
        return "THF"
    if lower == "dmf":
        return "DMF"
    if lower == "dmso":
        return "DMSO"
    if lower in {"chcl3", "chloroform"}:
        return "Chloroform"
    if lower in {"meoh", "methanol"}:
        return "Methanol"
    if lower in {"etoh", "ethanol"}:
        return "Ethanol"
    if lower in {"ipa", "isopropanol", "2-propanol"}:
        return "Isopropanol"
    if lower in {"acn", "acetonitrile"}:
        return "Acetonitrile"
    return n.title()


def _normalize_solvent(sol: Dict[str, Any]) -> None:
    solvent = sol.get("solvent")
    if solvent:
        s = normalize_for_parsing(str(solvent)).lower()
        if "methyl ethyl ketone" in s:
            s = s.replace("methyl ethyl ketone", "butanone")
        if "butanone" in s and "cyclohexane" in s:
            sol["solvent"] = "Butanone / Cyclohexane"
        elif "acetone" in s and "water" in s:
            sol["solvent"] = "Acetone / Water"
        elif "tetrahydrofuran" in s or s.strip() == "thf":
            sol["solvent"] = "THF"
        elif "chloroform" in s or "chcl3" in s:
            sol["solvent"] = "Chloroform"
        elif "/" in s or " and " in s:
            parts = re.split(r"\s*/\s*|\s+and\s+", s, maxsplit=1)
            if len(parts) >= 2:
                sol["solvent"] = f"{_format_solvent_name(parts[0])} / {_format_solvent_name(parts[1])}"
    ratio_units = sol.get("ratio_units")
    if ratio_units:
        ru = str(ratio_units).lower()
        if (
            ("wt%" in ru or "w/w" in ru or ("wt" in ru and "%" in ru))
            and "acetone" not in ru
            and "acetone" in str(sol.get("solvent") or "").lower()
        ):
            sol["ratio_units"] = "wt% acetone"
        if ru == "%" and "acetone" in str(sol.get("solvent") or "").lower():
            sol["ratio_units"] = "wt% acetone"
        if "vol" in ru or "v/v" in ru:
            sol["ratio_units"] = "vol%"


def _merge_tech(existing_tech: Dict[str, Any], new_tech: Dict[str, Any]) -> None:
    _merge_group(existing_tech, new_tech, TECH_FIELDS)
    for field in ("temperature", "flow_rate", "injection_volume"):
        if existing_tech.get(field) and new_tech.get(field) and existing_tech.get(field) != new_tech.get(field):
            merged = _merge_numeric_range(existing_tech.get(field), new_tech.get(field))
            if merged:
                existing_tech[field] = merged


def _condition_key(cond: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    unit = _norm(cond.get("critical_polymer_unit"))
    sp = cond.get("SP_FIELDS") or {}
    sol = cond.get("SOL_FIELDS") or {}
    key = (
        unit,
        _norm(sp.get("column_name")),
        _norm(sp.get("manufacturer")),
        _norm(sp.get("column_dimensions")),
        _norm(sp.get("phase")),
        _norm(sol.get("solvent")),
    )
    if all(not k for k in key[1:]):
        snippet = _norm(cond.get("context_snippet") or "")
        if snippet:
            return (unit, snippet, "", "", "", "")
    return key


def _infer_phase(sp: Dict[str, Any]) -> None:
    if not sp or sp.get("phase"):
        return
    name = _norm(sp.get("column_name"))
    mod = _norm(sp.get("modification"))
    material = _norm(sp.get("material"))

    if "diol" in name or "diol" in mod or "diol" in material:
        sp["phase"] = "Normal phase"
        return
    if "dvb" in name or "divinyl" in material:
        sp["phase"] = "Reversed phase (polymeric)"
        return
    if "hs-peg" in name or ("peg" in name and "silica" in material) or "peg" in mod:
        sp["phase"] = "Reversed phase (PEG-modified silica)"
        return
    if "rp" in name:
        sp["phase"] = "Reversed phase"


def _normalize_sp_units(sp: Dict[str, Any]) -> None:
    if not sp:
        return
    pd = sp.get("particle_diameter")
    if isinstance(pd, (int, float)):
        sp["particle_diameter"] = f"{pd:g} um"
    ps = sp.get("pore_size")
    if isinstance(ps, (int, float)):
        sp["pore_size"] = f"{ps:g} A"


def _normalize_ratio(sol: Dict[str, Any]) -> None:
    if not sol:
        return
    ratio = sol.get("ratio")
    if ratio is None:
        return
    if isinstance(ratio, (int, float)):
        sol["ratio"] = f"{ratio:g}"
        return
    if not isinstance(ratio, str):
        return
    r = ratio.strip()
    m = re.search(r"(wt%|w/w|v/v|vol%)", r, re.I)
    if m and not sol.get("ratio_units"):
        unit = m.group(1).lower()
        if unit == "w/w":
            unit = "wt%"
        if unit == "v/v":
            unit = "vol%"
        sol["ratio_units"] = unit
    r = re.sub(r"\s*(wt%|w/w|v/v|vol%)", "", r, flags=re.I).strip()
    r = re.sub(r"\s+", "", r)
    if r:
        sol["ratio"] = r


def _infer_column_name_from_snippets(snippets: List[str]) -> str | None:
    if not snippets:
        return None
    text = normalize_for_parsing(" ".join(snippets)).lower()
    if "nucleosil" in text or "nucleosol" in text:
        if "c18" in text or "c 18" in text:
            return "Nucleosil C18"
        return "Nucleosil"
    if "discovery" in text and ("hs-peg" in text or "hs peg" in text or "peg" in text):
        return "Discovery HS-PEG"
    if "jordi" in text and "dvb" in text:
        return "Jordi Gel DVB"
    if "hs-peg" in text or "hs peg" in text:
        return "HS-PEG"
    if "jordi" in text:
        return "Jordi"
    if "styragel" in text:
        return "Styragel"
    if "plgel" in text:
        return "PLgel"
    # Generic brand + C18
    m = re.search(r"\b([a-z][a-z0-9\-]+)\s+c\s*18\b", text)
    if m:
        brand = m.group(1)
        return f"{brand.title()} C18"
    return None


def _normalize_column_name(sp: Dict[str, Any]) -> None:
    if not sp:
        return
    name = sp.get("column_name")
    if not name:
        return
    n = normalize_for_parsing(str(name)).strip()
    low = n.lower()
    if "nucleosol" in low:
        n = re.sub(r"nucleosol", "Nucleosil", n, flags=re.I)
    if "hs peg" in low:
        n = re.sub(r"hs\\s*peg", "HS-PEG", n, flags=re.I)
    if low.startswith("of c18") or low.startswith("of c 18"):
        n = re.sub(r"^of\\s+", "", n, flags=re.I)
    if "c18" in low and "nucleosil" in low:
        n = "Nucleosil C18"
    sp["column_name"] = n


def _ratio_candidates_from_snippets(snippets: List[str], aliases: List[str]) -> List[float]:
    candidates: List[float] = []
    fallback_candidates: List[float] = []
    for snippet in snippets:
        if not snippet:
            continue
        text = normalize_for_parsing(str(snippet)).lower()
        sentences = re.split(r"(?<!\d)[.!?](?!\d)", text)
        for sent in sentences:
            if not re.search(r"(wt%|w/w|%|v/v|vol%)", sent):
                continue
            if aliases and not any(alias in sent for alias in aliases):
                if "this column" in sent or "the column" in sent:
                    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:wt%|w/w|%)", sent):
                        try:
                            fallback_candidates.append(float(match.group(1)))
                        except Exception:
                            continue
                continue
            for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:wt%|w/w|%)", sent):
                try:
                    candidates.append(float(match.group(1)))
                except Exception:
                    continue
    # Deduplicate with rounding to avoid float noise
    if fallback_candidates:
        candidates.extend(fallback_candidates)
    uniq = {round(val, 4) for val in candidates}
    return sorted(uniq)


def _ratio_pair_from_snippets(snippets: List[str]) -> Tuple[str, str] | None:
    pattern = re.compile(r"(\d+(?:\.\d+)?)\s*[/:\-]\s*(\d+(?:\.\d+)?)\s*(vol%|v/v|%|vol)", re.I)
    for snippet in snippets:
        if not snippet:
            continue
        text = normalize_for_parsing(str(snippet)).lower()
        m = pattern.search(text)
        if not m:
            continue
        ratio = f"{m.group(1)}/{m.group(2)}"
        units = m.group(3).lower()
        if units == "vol":
            units = "vol%"
        return ratio, units
    return None


def _clean_tech(tech: Dict[str, Any]) -> None:
    if not tech:
        return
    inj = tech.get("injection_volume")
    if inj is None:
        return
    inj_text = str(inj).lower()
    if ("wt%" in inj_text or "%" in inj_text) and "ul" not in inj_text and "ml" not in inj_text:
        tech["injection_volume"] = None


def link_conditions(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge and dedupe LCCC conditions."""
    merged: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}

    def _extract_ratio(solvent: str) -> Tuple[str, str]:
        if not solvent:
            return "", ""
        s = normalize_for_parsing(str(solvent))
        m = re.search(r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*(wt%|w/w|v/v|%)", s, re.I)
        if not m:
            m2 = re.search(
                r"(\d+(?:\.\d+)?)\s*[/:\-]\s*(\d+(?:\.\d+)?)\s*(vol%|v/v|%|vol)",
                s,
                re.I,
            )
            if not m2:
                return "", ""
            ratio = f"{m2.group(1)}/{m2.group(2)}"
            units = m2.group(3).lower()
            if units == "vol":
                units = "vol%"
            return ratio, units
        ratio = re.sub(r"\s+", "", m.group(1))
        units = m.group(2).lower()
        return ratio, units

    def _merge_solvent(existing_sol: Dict[str, Any], new_sol: Dict[str, Any]) -> None:
        _merge_group(existing_sol, new_sol, SOL_FIELDS)
        if existing_sol.get("ratio") and new_sol.get("ratio") and existing_sol.get("ratio") != new_sol.get("ratio"):
            a = _parse_ratio_range(existing_sol.get("ratio"))
            b = _parse_ratio_range(new_sol.get("ratio"))
            if a and b:
                lo = min(a[0], b[0])
                hi = max(a[1], b[1])
                existing_sol["ratio"] = f"{lo:g}-{hi:g}" if lo != hi else f"{lo:g}"

        if existing_sol.get("ratio_units") and new_sol.get("ratio_units"):
            if existing_sol["ratio_units"] != new_sol["ratio_units"]:
                # prefer wt% over generic % when both appear
                if "wt%" in str(new_sol["ratio_units"]).lower():
                    existing_sol["ratio_units"] = new_sol["ratio_units"]

        _normalize_solvent(existing_sol)

    for cond in conditions:
        sol = cond.get("SOL_FIELDS") or {}
        if sol.get("solvent") and sol.get("ratio") is None:
            ratio, units = _extract_ratio(str(sol.get("solvent")))
            if ratio:
                sol["ratio"] = ratio
            if units and not sol.get("ratio_units"):
                sol["ratio_units"] = units
            cond["SOL_FIELDS"] = sol
        _normalize_solvent(sol)

        key = _condition_key(cond)
        if key in merged:
            existing = merged[key]
            _merge_group(existing.get("SP_FIELDS", {}), cond.get("SP_FIELDS", {}), SP_FIELDS)
            _merge_solvent(existing.get("SOL_FIELDS", {}), cond.get("SOL_FIELDS", {}))
            _merge_tech(existing.get("TECH_FIELDS", {}), cond.get("TECH_FIELDS", {}))
            _merge_group(existing.get("separation_behavior", {}), cond.get("separation_behavior", {}), SEP_FIELDS)

            if cond.get("context_snippet"):
                existing.setdefault("context_snippets", []).append(cond.get("context_snippet"))
            continue

        merged[key] = {
            "critical_polymer_unit": cond.get("critical_polymer_unit"),
            "SP_FIELDS": cond.get("SP_FIELDS", {}),
            "SOL_FIELDS": cond.get("SOL_FIELDS", {}),
            "TECH_FIELDS": cond.get("TECH_FIELDS", {}),
            "separation_behavior": cond.get("separation_behavior", {}),
        }
        if cond.get("context_snippet"):
            merged[key]["context_snippets"] = [cond.get("context_snippet")]

    results = list(merged.values())
    for cond in results:
        sp = cond.get("SP_FIELDS") or {}
        sol = cond.get("SOL_FIELDS") or {}
        tech = cond.get("TECH_FIELDS") or {}
        name = _norm(sp.get("column_name"))

        # Improve ratio ranges using context snippets tied to the column name
        aliases: List[str] = []
        extra_text = " ".join(
            [
                name,
                _norm(sp.get("material")),
                _norm(sp.get("modification")),
                _norm(sp.get("phase")),
            ]
        )
        for token in ("jordi", "discovery", "hs-peg", "nucleosil", "diol"):
            if token in extra_text:
                aliases.append(token)
        if not aliases and name:
            aliases = [t for t in re.split(r"[^a-z0-9]+", name) if len(t) >= 4]

        snippets = cond.get("context_snippets") or []
        if not sp.get("column_name"):
            inferred = _infer_column_name_from_snippets(snippets)
            if inferred:
                sp["column_name"] = inferred
        _normalize_column_name(sp)

        if not sol.get("ratio"):
            pair = _ratio_pair_from_snippets(snippets)
            if pair:
                sol["ratio"] = pair[0]
                if not sol.get("ratio_units"):
                    sol["ratio_units"] = pair[1]

        candidates = _ratio_candidates_from_snippets(snippets, aliases)
        if candidates and (not sol.get("ratio") or _parse_ratio_range(sol.get("ratio")) is not None):
            current = sol.get("ratio")
            current_range = _parse_ratio_range(current)
            if current_range is None or (len(candidates) > 1):
                lo = min(candidates)
                hi = max(candidates)
                sol["ratio"] = f"{lo:g}-{hi:g}" if lo != hi else f"{lo:g}"
            if not sol.get("ratio_units") and sol.get("solvent") and "acetone" in str(sol.get("solvent")).lower():
                sol["ratio_units"] = "wt% acetone"

        _normalize_solvent(sol)
        _infer_phase(sp)
        _normalize_sp_units(sp)
        _normalize_ratio(sol)
        _clean_tech(tech)

        cond["SP_FIELDS"] = sp
        cond["SOL_FIELDS"] = sol
        cond["TECH_FIELDS"] = tech
    return results


def validate_conditions(data: List[Dict[str, Any]]) -> bool:
    try:
        if not isinstance(data, list):
            return False
        for cond in data:
            if not isinstance(cond, dict):
                return False
            if "SP_FIELDS" not in cond or "SOL_FIELDS" not in cond or "TECH_FIELDS" not in cond:
                return False
        return True
    except Exception:
        return False
