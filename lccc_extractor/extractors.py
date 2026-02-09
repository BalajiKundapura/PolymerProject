import json
import re
from typing import Dict, List, Optional, Set, Tuple

from .logging_config import logger
from .models import SolventDetail, StationaryPhase, TechnicalDetail
from .normalization import normalize_for_parsing, normalize_value
from .patterns import (
    COLUMN_PHASE_TERMS,
    RE_CONC,
    RE_DIM,
    RE_FLOW,
    RE_INJECT_SOL,
    RE_INJECT_VOL,
    RE_MANUF_LINE,
    RE_NUM_COLS,
    RE_PARTICLE,
    RE_PORE,
    RE_SOLVENT_COMPONENT,
    RE_SOLVENT_COMPONENT_PAREN,
    RE_SOLVENT_MIX,
    RE_SOLVENT_MIXTURE,
    RE_SOLVENT_MIXTURE_OF,
    RE_SOLVENT_MIXTURE_SLASH,
    RE_SOLVENT_PAREN,
    RE_TEMPERATURE,
)


def _has_stationary_phase_data(sp: StationaryPhase) -> bool:
    return any(getattr(sp, k) is not None for k in sp.__dataclass_fields__)


def _has_technical_data(td: TechnicalDetail) -> bool:
    return any(getattr(td, k) is not None for k in td.__dataclass_fields__)


def _dedup_stationary_phases(phases: List[StationaryPhase]) -> List[StationaryPhase]:
    seen: Set[str] = set()
    out: List[StationaryPhase] = []
    for sp in phases:
        key = json.dumps(sp.to_dict(), sort_keys=True, ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(sp)
    return out


def _extract_stationary_phase(text: str) -> StationaryPhase:
    sp = StationaryPhase()
    try:
        t = normalize_for_parsing(text)
        t_l = t.lower()

        # Quick gate: if there's no column-ish language, skip to avoid false positives.
        columnish = bool(re.search(r"\bcolumns?\b|\bcol\.\b|stationary phase|packed with", t_l))
        if not columnish:
            return sp

        # Avoid instrument-only phrases like "column oven" in system descriptions.
        if "column oven" in t_l and not re.search(r"\b(?:c18|rp[- ]?18|diol|amino|ods|silica)\b", t_l):
            if not (RE_DIM.search(t) or RE_PORE.search(t) or RE_PARTICLE.search(t) or RE_MANUF_LINE.search(t)):
                return sp

        def _clean_column_ref(raw: str) -> str:
            s = (raw or "").strip()
            if not s:
                return s
            tokens = s.split()
            stop = {"the", "a", "an", "system", "column", "columns", "col", "col.", "on", "in", "with", "using", "via"}
            sig_idx = None
            for i, tok in enumerate(tokens):
                tok_l = tok.lower().strip(".")
                if tok_l in stop:
                    continue
                if re.search(r"[0-9]", tok) or re.search(r"[A-Z]", tok):
                    sig_idx = i
                    break
            if sig_idx is not None and sig_idx > 0:
                s = " ".join(tokens[sig_idx:])
            s = re.sub(r"^(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b\s+", "", s, flags=re.I)
            s = re.sub(r"^(?:reversed\s+phase|normal\s+phase)\b\s+", "", s, flags=re.I)
            s = re.sub(r"^(?:on|in|with|using|via)\b\s+", "", s, flags=re.I)
            return re.sub(r"\s+", " ", s).strip()

        # Prefer "Name column(s)" patterns (captures name BEFORE the word column/columns).
        ref_candidates: List[str] = []
        for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9\- ]{1,60})\s+(?:columns?|col\.)\b", t, re.I):
            name = _clean_column_ref(m.group(1))
            if not name:
                continue
            if name.lower() in {"a", "an", "the", "this", "that", "another", "other"}:
                continue
            # Heuristic: true column names usually contain uppercase letters, digits, or hyphens.
            if not re.search(r"[A-Z0-9\-]", name):
                continue
            # Filter out obvious non-column phrases.
            if any(bad in name.lower() for bad in ("oven", "system", "selection", "detector", "pump", "autosampler")):
                continue
            ref_candidates.append(name)

        if ref_candidates:
            best = max(ref_candidates, key=len).strip()
            # Avoid storing generic shorthand as a "column name" (use phase instead).
            if best.lower() not in {"rp", "rp18", "rp-18", "c18", "ods", "cn", "cyano"}:
                sp.column_name = normalize_value(best)

        m_pore = RE_PORE.search(t)
        if m_pore:
            sp.pore_size = normalize_value(f"{m_pore.group(1)} {m_pore.group(2)}")

        m_part = RE_PARTICLE.search(t)
        if m_part:
            unit = m_part.group(2) or ""
            if unit.lower() == "mm":
                try:
                    if float(m_part.group(1)) <= 50:
                        unit = "um"
                except Exception:
                    pass
            sp.particle_diameter = normalize_value(f"{m_part.group(1)} {unit}".strip())

        # Common column codes like "100-5" encode pore size + particle size in many LC columns:
        #   100-5 => 100 A pore size, 5 um particles
        # When multiple columns are connected, this may appear as a set: (100-5, 300-5, 1000-7).
        if sp.pore_size is None or sp.particle_diameter is None:
            pore_vals: List[int] = []
            part_vals: List[int] = []
            for a, b in re.findall(r"\b(\d{2,4})\s*-\s*(\d{1,2})\b", t):
                try:
                    pore = int(a)
                    part = int(b)
                except Exception:
                    continue
                if not (40 <= pore <= 5000 and 1 <= part <= 20):
                    continue
                pore_vals.append(pore)
                part_vals.append(part)

            if pore_vals and sp.pore_size is None:
                unique = sorted(set(pore_vals))
                sp.pore_size = normalize_value(("/".join(str(x) for x in unique) + " A") if len(unique) > 1 else f"{unique[0]} A")
            if part_vals and sp.particle_diameter is None:
                unique = sorted(set(part_vals))
                sp.particle_diameter = normalize_value(("/".join(str(x) for x in unique) + " um") if len(unique) > 1 else f"{unique[0]} um")

        # Some papers list particle size as a bare unit near column specs, e.g. "SDV 5 um precolumn".
        if sp.particle_diameter is None:
            m_bare = re.search(r"\b(\d+(?:\.\d+)?)\s*(um|nm)\b", t, re.I)
            if m_bare and re.search(r"\b(?:particle|precolumn|columns?|packing|packed|sdv|nucleosil|silica|c18)\b", t_l):
                sp.particle_diameter = normalize_value(f"{m_bare.group(1)} {m_bare.group(2)}")

        # Some papers list pore size as a bare Angstrom unit near column specs, e.g. "100 A".
        if sp.pore_size is None:
            m_bare = re.search(r"\b(\d{2,5})\s*A\b", t)
            if m_bare and re.search(r"\b(?:pore|columns?|packing|packed|sdv|nucleosil|silica|c18)\b", t_l):
                sp.pore_size = normalize_value(f"{m_bare.group(1)} A")

        m_dim = RE_DIM.search(t)
        if m_dim:
            sp.column_dimensions = normalize_value(f"{m_dim.group(1)}x{m_dim.group(2)} {m_dim.group(3)}")

        m_num = RE_NUM_COLS.search(t)
        if m_num:
            sp.number_of_columns = normalize_value(m_num.group(1))
        else:
            # e.g. "three reversed phase ... columns"
            m_num2 = re.search(
                r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b(?:\s+[A-Za-z0-9\-]{1,15}){0,6}\s+columns?\b",
                t,
                re.I,
            )
            if m_num2:
                # Avoid treating "C 18 columns" (PDF spacing for C18) as "18 columns"
                start = m_num2.start(1)
                prefix = t[max(0, start - 3):start].lower()
                if re.search(r"\bc\s*$", prefix) or re.search(r"\brp\s*$", prefix):
                    m_num2 = None
            if m_num2:
                word = m_num2.group(1).lower()
                word_map = {
                    "one": "1",
                    "two": "2",
                    "three": "3",
                    "four": "4",
                    "five": "5",
                    "six": "6",
                    "seven": "7",
                    "eight": "8",
                    "nine": "9",
                    "ten": "10",
                }
                sp.number_of_columns = word_map.get(word, word)

        # Try to pick a column manufacturer (avoid instrument vendor lines like "HPLC system").
        manuf_candidates: List[str] = []
        for m_man in RE_MANUF_LINE.finditer(t):
            cand = m_man.group(0).strip()
            cand_l = cand.lower()
            if any(bad in cand_l for bad in ("hplc system", "pump", "detector", "autosampler", "software", "spectros")):
                continue
            manuf_candidates.append(cand)
        if manuf_candidates:
            sp.manufacturer = normalize_value(manuf_candidates[-1])

        phase = None
        # Explicit "stationary phase RP-18 ..." pattern
        st_m = re.search(r"\bstationary\s+phase\s+([A-Za-z0-9\-]{2,20})\b", t, re.I)
        if st_m:
            phase = st_m.group(1)

        phase_m = re.search(r"\b([A-Za-z0-9\-]{2,20})\s+phase\b", t, re.I)
        if phase_m and phase is None:
            cand = phase_m.group(1)
            if cand.lower() not in {"mobile", "stationary", "reversed", "normal"}:
                phase = cand

        if re.search(r"\brp[- ]?18\b", t_l) and phase is None:
            phase = "RP-18"

        for term in COLUMN_PHASE_TERMS:
            if term.lower() in t_l:
                phase = term
                break
        sp.phase = phase

        mat_m = re.search(r"\b(?:material|packed with)\s*[:=]?\s*([A-Za-z0-9% \-\(\)\/]+?)(?=;|\.|,|\)|\n)", t, re.I)
        pol_m = None
        if re.search(r"\b100%\s*poly\([^)]+\)", t, re.I):
            pol_m = re.search(r"\bpoly\([^)]+\)", t)
        elif "packed with" in t_l or "material" in t_l:
            pol_m = re.search(r"\bpoly\([^)]+\)", t)
        sp.material = normalize_value(pol_m.group(0) if pol_m else (mat_m.group(1) if mat_m else None))

        # If we have a "columns (...)" list, append it to the column_name for traceability.
        if sp.column_name:
            m_set = re.search(r"\bcolumns?\s*\(([^)]+)\)", t, re.I)
            if m_set and re.search(r"\d+\s*-\s*\d+", m_set.group(1)):
                appendix = re.sub(r"\s+", " ", m_set.group(1)).strip()
                if len(appendix) <= 80 and appendix.lower() not in sp.column_name.lower():
                    sp.column_name = normalize_value(f"{sp.column_name} ({appendix})")

        mod_m = re.search(r"\b(?:modified|grafted|bonded)\s*with\s*([A-Za-z0-9 \-\(\)\/]+?)(?=;|\.|,|\)|\n)", t, re.I)
        sp.modification = normalize_value(mod_m.group(1) if mod_m else None)
    except Exception as e:
        logger.warning(f"Error extracting stationary phase: {e}")

    return sp


def _extract_solvents(text: str) -> List[SolventDetail]:
    sols: List[SolventDetail] = []
    try:
        t = normalize_for_parsing(text)

        def is_solvent_context(start: int, end: int) -> bool:
            window = (t[max(0, start - 80):start] + " " + t[end:end + 80]).lower()
            return any(k in window for k in ("mobile phase", "eluent", "solvent", "mixture", "dissolved in", "mobile phases"))

        def looks_like_solvent_name(val: str) -> bool:
            v = val.strip()
            if not v or len(v) > 60:
                return False
            v_l = v.lower()
            if v_l in {"poly", "polymer"} or v_l.startswith("poly("):
                return False
            # Prevent obvious non-solvents / common false positives in polymer chromatography text.
            if v_l in {"on-flow", "onflow", "off-line", "offline", "on-line", "online"}:
                return False
            if any(k in v_l for k in ("macherey", "agilent", "waters", "phenomenex", "shimadzu", "thermo", "supelco", "tosoh", "jordi", "chromtech")):
                return False
            if v_l in {"composition", "behavior", "function", "mode", "conditions", "diagram"}:
                return False
            if any(bad in v_l for bad in ("critical", "conditions", "reported", "observed", "found", "column")):
                return False
            if "-" in v:
                parts = [p.strip() for p in v.split("-")]
                if all(p.isupper() for p in parts):
                    return False
                return 2 <= len(parts) <= 4 and all(1 < len(p) <= 20 for p in parts)
            if "/" in v:
                parts = [p.strip() for p in v.split("/")]
                if len(parts) < 2:
                    return False
                # Avoid polymer block abbreviations like "EO/PO"
                if all(p.isupper() and len(p) <= 3 for p in parts):
                    return False
                unit_tokens = {"l", "ml", "ul", "min", "minute", "h", "hr", "s", "sec", "g", "mg", "ug", "kg", "mol"}
                parts_l = [p.lower() for p in parts]
                if all(p in unit_tokens for p in parts_l):
                    return False
                return all(1 < len(p) <= 20 for p in parts)
            # allow up to 2 short words (e.g., "ethyl acetate")
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*)?", v):
                return True
            return False

        def component_in_mix(component: str, mix: str) -> bool:
            comp = component.strip().lower()
            mix_l = mix.lower()
            parts = [p.strip() for p in re.split(r"[/-]", mix_l) if p.strip()]
            if not parts:
                return False
            return any(comp == p or comp in p or p in comp for p in parts)

        for m in RE_SOLVENT_MIX.finditer(t):
            solvent = re.sub(r"^(?:in|on|with|and)\s+", "", m.group(1).strip(), flags=re.I)
            if not looks_like_solvent_name(solvent):
                continue
            sols.append(
                SolventDetail(
                    solvent=normalize_value(solvent),
                    ratio=normalize_value(m.group(2)) if m.group(2) else None,
                    ratio_units=normalize_value(m.group(3)) if m.group(3) else None,
                )
            )
        for m in RE_SOLVENT_PAREN.finditer(t):
            solvent = re.sub(r"^(?:in|on|with|and)\s+", "", m.group(1).strip(), flags=re.I)
            if not looks_like_solvent_name(solvent):
                continue
            sols.append(SolventDetail(solvent=normalize_value(solvent), ratio=normalize_value(m.group(2)), ratio_units=normalize_value(m.group(3))))

        # Handle patterns like "THF was used as mobile phase" or "using THF as mobile phase"
        def _clean_solvent_phrase(s: str) -> str:
            s = s.strip(" ,;:.")
            # Drop common grade/descriptive prefixes
            s = re.sub(r"\b(?:hplc|uhplc)\s+grade\b", "", s, flags=re.I)
            s = re.sub(r"\bgrade\b", "", s, flags=re.I)
            s = re.sub(r"\b(?:protonated|deuterated|anhydrous|dry)\b", "", s, flags=re.I)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        for m in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,2})\s+(?:was|were|is|are)\s+used\s+as\s+(?:the\s+)?(?:mobile\s*phase|eluent)\b",
            t,
            re.I,
        ):
            solvent = _clean_solvent_phrase(m.group(1))
            if solvent and looks_like_solvent_name(solvent):
                sols.append(SolventDetail(solvent=normalize_value(solvent), ratio=None, ratio_units=None))

        for m in re.finditer(
            r"\b(?:using|with)\s+([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z][A-Za-z0-9\-]*){0,2})\s+as\s+(?:the\s+)?(?:mobile\s*phase|eluent)\b",
            t,
            re.I,
        ):
            solvent = _clean_solvent_phrase(m.group(1))
            if solvent and looks_like_solvent_name(solvent):
                sols.append(SolventDetail(solvent=normalize_value(solvent), ratio=None, ratio_units=None))

        # Capture mixture names that appear anywhere (used for attaching component ratios).
        mixtures: List[Tuple[int, int, str]] = []
        for m in RE_SOLVENT_MIXTURE.finditer(t):
            mix = m.group(1).strip()
            if not looks_like_solvent_name(mix):
                continue
            if not is_solvent_context(m.start(), m.end()) and "water" not in mix.lower():
                continue
            mixtures.append((m.start(), m.end(), mix))
        for m in RE_SOLVENT_MIXTURE_SLASH.finditer(t):
            mix = re.sub(r"\s*/\s*", "/", m.group(1))
            if not looks_like_solvent_name(mix):
                continue
            # Require an explicit solvent cue OR a nearby ratio like "(92/8 vol%)"
            if not is_solvent_context(m.start(), m.end()) and "water" not in mix.lower():
                tail = t[m.end():m.end() + 60]
                if not re.search(r"\(\s*\d+(?:\.\d+)?\s*[:/]\s*\d+", tail):
                    continue
            mixtures.append((m.start(), m.end(), mix))
        for m in RE_SOLVENT_MIXTURE_OF.finditer(t):
            mix = f"{m.group(1)}/{m.group(2)}"
            if looks_like_solvent_name(mix):
                mixtures.append((m.start(), m.end(), mix))

        mixtures.sort(key=lambda x: x[0])

        # Add mixture names even when ratios are not specified (only when in solvent context).
        for _, __, mix in mixtures:
            sols.append(SolventDetail(solvent=normalize_value(mix), ratio=None, ratio_units=None))

        # Extract ratio lists like "butanone/cyclohexane: 70:30; 92:8; 97:3"
        for m_start, m_end, mix in mixtures:
            tail = t[m_end:m_end + 220]
            for rm in re.finditer(r"\b(\d+(?:\.\d+)?)([:/])(\d+(?:\.\d+)?)(?:\s*(vol%|wt%|v/v|w/w|%))?", tail, re.I):
                a = rm.group(1)
                delim = rm.group(2)
                b = rm.group(3)
                units = rm.group(4)
                delim_norm = "/" if delim == ":" else delim
                ratio = f"{a}{delim_norm}{b}"
                sols.append(
                    SolventDetail(solvent=normalize_value(mix), ratio=normalize_value(ratio), ratio_units=normalize_value(units) if units else None)
                )

        for m in RE_SOLVENT_COMPONENT.finditer(t):
            ratio = m.group(1)
            units = m.group(2)
            component = m.group(3)
            if component.lower() in {"poly", "polymer"} or component.lower().startswith("poly"):
                continue

            # If a mixture appears shortly before, attach component to the mixture for context.
            mix = None
            for pos, end, mix_name in reversed(mixtures):
                if pos < m.start() and (m.start() - end) <= 120:
                    mix = mix_name
                    break

            if mix and not component_in_mix(component, mix):
                mix = None

            solvent = f"{mix} ({component})" if mix else component

            if looks_like_solvent_name(solvent.replace(f" ({component})", "")) or mix:
                sols.append(SolventDetail(solvent=normalize_value(solvent), ratio=normalize_value(ratio), ratio_units=normalize_value(units)))

        # Handle percent with explicit ratio type: "45% (w/w) acetone"
        for m in RE_SOLVENT_COMPONENT_PAREN.finditer(t):
            ratio = m.group(1)
            units = m.group(2)
            component = m.group(3)
            if component.lower() in {"poly", "polymer"} or component.lower().startswith("poly"):
                continue

            mix = None
            for pos, end, mix_name in reversed(mixtures):
                if pos < m.start() and (m.start() - end) <= 120:
                    mix = mix_name
                    break

            if mix and not component_in_mix(component, mix):
                mix = None

            solvent = f"{mix} ({component})" if mix else component
            if looks_like_solvent_name(solvent.replace(f" ({component})", "")) or mix:
                sols.append(SolventDetail(solvent=normalize_value(solvent), ratio=normalize_value(ratio), ratio_units=normalize_value(units)))

        dedup: Dict[Tuple[str, str, str], SolventDetail] = {}
        for s in sols:
            key = (s.solvent or "", s.ratio or "", s.ratio_units or "")
            dedup[key] = s
        sols = list(dedup.values())
    except Exception as e:
        logger.warning(f"Error extracting solvents: {e}")

    return sols


def _extract_technical(text: str) -> TechnicalDetail:
    td = TechnicalDetail()
    try:
        t = normalize_for_parsing(text)
        mT = RE_TEMPERATURE.search(t)
        if mT:
            td.temperature = normalize_value(f"{mT.group(1)} {mT.group(2)}")
        mF = RE_FLOW.search(t)
        if mF:
            td.flow_rate = normalize_value(f"{mF.group(1)} {mF.group(2)}/{mF.group(3)}")
        mC = RE_CONC.search(t)
        if mC:
            td.injected_polymer_concentration = normalize_value(f"{mC.group(1)} {mC.group(2)}")
        mS = RE_INJECT_SOL.search(t)
        if mS:
            td.injected_polymer_solvent_solution = normalize_value(mS.group(1))
        mV = RE_INJECT_VOL.search(t)
        if mV:
            td.injection_volume = normalize_value(f"{mV.group(1)} {mV.group(2)}")
    except Exception as e:
        logger.warning(f"Error extracting technical details: {e}")
    return td


def _merge_technical(base: TechnicalDetail, override: TechnicalDetail) -> TechnicalDetail:
    return TechnicalDetail(
        temperature=override.temperature or base.temperature,
        flow_rate=override.flow_rate or base.flow_rate,
        injected_polymer_concentration=override.injected_polymer_concentration or base.injected_polymer_concentration,
        injected_polymer_solvent_solution=override.injected_polymer_solvent_solution or base.injected_polymer_solvent_solution,
        injection_volume=override.injection_volume or base.injection_volume,
    )


def _merge_technical_fill(base: TechnicalDetail, add: TechnicalDetail) -> TechnicalDetail:
    """
    Fill-only merge: keep existing values and only populate missing fields.
    Useful when building a global context where later paragraphs may contain
    other temperatures (e.g., detector evaporator) that should not override
    the main column temperature.
    """
    return TechnicalDetail(
        temperature=base.temperature or add.temperature,
        flow_rate=base.flow_rate or add.flow_rate,
        injected_polymer_concentration=base.injected_polymer_concentration or add.injected_polymer_concentration,
        injected_polymer_solvent_solution=base.injected_polymer_solvent_solution or add.injected_polymer_solvent_solution,
        injection_volume=base.injection_volume or add.injection_volume,
    )


def _extract_stationary_phases(text: str) -> List[StationaryPhase]:
    """
    Extract one or more stationary phase / column definitions from a paragraph.
    Handles both "X column" references and bullet/list style specifications.
    """
    t = normalize_for_parsing(text)
    phases: List[StationaryPhase] = []

    found_list_style = False

    # Detect list-style blocks: "The following columns were used ...: <spec>. <spec>. <spec>."
    if re.search(r"\bfollowing\s+columns\b|\bcolumns\s+were\s+used\b", t, re.I) and ":" in t:
        _, rest = t.split(":", 1)
        # Column specs in PDF text are frequently separated by "). <NextSpec>".
        entries = [e.strip(" .;") for e in re.split(r"\.\s*(?=[A-Z])", rest) if e.strip()]
        for entry in entries:
            sp = StationaryPhase()
            entry = entry.strip()

            # Column name typically appears at the beginning up to ":" or "," or ";".
            name = None
            sep_candidates: List[Tuple[int, str]] = []
            for sep in (":", ",", ";"):
                pos = entry.find(sep)
                if pos != -1 and pos <= 60:
                    sep_candidates.append((pos, sep))
            if sep_candidates:
                _, sep = sorted(sep_candidates, key=lambda x: x[0])[0]
                cand = entry.split(sep, 1)[0].strip()
                if 3 <= len(cand) <= 80:
                    name = cand
            if not name:
                # fallback: take first 6 tokens
                tokens = entry.split()
                name = " ".join(tokens[:6]).strip() if tokens else None

            if name:
                sp.column_name = normalize_value(name)

            # Parse known attributes from entry text
            m_dim = RE_DIM.search(entry)
            if m_dim:
                sp.column_dimensions = normalize_value(f"{m_dim.group(1)}x{m_dim.group(2)} {m_dim.group(3)}")

            m_part = RE_PARTICLE.search(entry)
            if m_part:
                unit = m_part.group(2) or ""
                if unit.lower() == "mm":
                    try:
                        if float(m_part.group(1)) <= 50:
                            unit = "um"
                    except Exception:
                        pass
                sp.particle_diameter = normalize_value(f"{m_part.group(1)} {unit}".strip())

            m_pore = RE_PORE.search(entry)
            if m_pore:
                sp.pore_size = normalize_value(f"{m_pore.group(1)} {m_pore.group(2)}")

            pol_m = re.search(r"\bpoly\([^)]+\)", entry)
            if pol_m:
                sp.material = normalize_value(pol_m.group(0))

            phase_m = re.search(r"\b([A-Za-z0-9\-]{2,20})\s+phase\b", entry, re.I)
            if phase_m:
                sp.phase = normalize_value(phase_m.group(1))

            parens = re.findall(r"\(([^)]+)\)", entry)
            manuf_candidates = [p.strip() for p in parens if "," in p]
            if manuf_candidates:
                sp.manufacturer = normalize_value(manuf_candidates[-1])

            if _has_stationary_phase_data(sp):
                phases.append(sp)
                found_list_style = True

    # Generic single-column extraction from any paragraph
    if not found_list_style:
        sp = _extract_stationary_phase(t)
        if _has_stationary_phase_data(sp):
            phases.append(sp)

    return _dedup_stationary_phases(phases)


def _match_stationary_phases(text: str, catalog: List[StationaryPhase]) -> List[StationaryPhase]:
    """
    Try to select the most relevant column(s) from a catalog based on paragraph text.
    """
    if not catalog:
        return []

    t = normalize_for_parsing(text).lower()
    scored: List[Tuple[int, StationaryPhase]] = []
    for sp in catalog:
        score = 0
        name = (sp.column_name or "").strip()
        manuf = (sp.manufacturer or "").strip()
        phase = (sp.phase or "").strip()

        name_l = name.lower()
        if name_l:
            # Avoid spurious substring matches for very short names like "RP" (matches "terpolymers").
            if len(name_l) <= 3:
                if re.search(rf"\b{re.escape(name_l)}\b", t):
                    score += 100
            else:
                if name_l in t:
                    score += 100

        # Brand-only references like "Jordi column"
        brand = manuf.split(",", 1)[0].strip().lower() if manuf else ""
        if brand and brand in t and "column" in t:
            score += 30

        # Weak signal: "Diol column", "C18 column"
        if phase and phase.lower() in t and "column" in t:
            score += 25

        # Special-case common RP-18/C18 shorthand
        if re.search(r"\b(?:c18|rp[- ]?18)\b", t):
            phase_l = phase.lower()
            if "c18" in phase_l or "rp-18" in phase_l or "rp18" in phase_l:
                score += 60
            elif "c 18" in name_l or "c18" in name_l:
                score += 60

        if score > 0:
            scored.append((score, sp))

    if not scored:
        return []

    best = max(s for s, _ in scored)
    # Require at least a medium-confidence match.
    if best < 30:
        return []

    winners = [sp for s, sp in scored if s == best]
    return _dedup_stationary_phases(winners)
