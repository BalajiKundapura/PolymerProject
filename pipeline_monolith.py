import os
import re
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Set

from polymerSubject import (
    extract_polymers,
    validate_polymers,
    select_main_polymers,
    normalize_name,
    polymer_abbrev_dict
)

try:
    from sentence_transformers import SentenceTransformer
    _MODEL_NAME = os.getenv("LCCC_SENT_MODEL", "all-MiniLM-L6-v2")
except Exception:
    SentenceTransformer = None
    _MODEL_NAME = None

logger = logging.getLogger("lccc_context_aware")
_log_level = os.getenv("LCCC_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _log_level, logging.INFO))
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)

# -----------------------
# Data structures
# -----------------------
@dataclass
class PolymerMention:
    """Represents a single mention of a polymer in text"""
    polymer_name: str
    canonical_name: str
    position: int  # Character position in text
    context_before: str  # 200 chars before
    context_after: str   # 200 chars after
    full_context: str    # Full paragraph
    para_idx: int
    
    def get_context_window(self) -> str:
        """Get the full context window around this mention"""
        return f"{self.context_before} [{self.polymer_name}] {self.context_after}"

@dataclass
class StationaryPhase:
    column_name: Optional[str] = None
    material: Optional[str] = None
    modification: Optional[str] = None
    pore_size: Optional[str] = None
    particle_diameter: Optional[str] = None
    column_dimensions: Optional[str] = None
    number_of_columns: Optional[str] = None
    manufacturer: Optional[str] = None
    phase: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v if v not in ("", None) else None) for k, v in asdict(self).items()}

@dataclass
class SolventDetail:
    solvent: Optional[str] = None
    ratio: Optional[str] = None
    ratio_units: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v if v not in ("", None) else None) for k, v in asdict(self).items()}

@dataclass
class TechnicalDetail:
    temperature: Optional[str] = None
    flow_rate: Optional[str] = None
    injected_polymer_concentration: Optional[str] = None
    injected_polymer_solvent_solution: Optional[str] = None
    injection_volume: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v if v not in ("", None) else None) for k, v in asdict(self).items()}

# -----------------------
# Lexicons
# -----------------------
_LCCC_POS_CUES = [
    "liquid chromatography at the critical condition",
    "critical condition",
    "critical conditions",
    "LCCC",
    "chromatographically invisible",
    "critical adsorption point",
    "critical point",
]

_NON_LCCC_EXCLUDE = [
    "size exclusion chromatography",
    "SEC",
    "GPC",
    "gel permeation chromatography",
    "ion exchange chromatography",
    "IEC",
    "anion exchange",
    "cation exchange",
]

_COLUMN_PHASE_TERMS = [
    "C18",
    "ODS",
    "Diol",
    "Amino",
    "silica",
    "HILIC",
    "reversed phase",
    "normal phase",
    "grafted",
    "bonded",
]

# -----------------------
# Compiled Regex Patterns
# -----------------------
RE_TEMPERATURE = re.compile(
    r"\b(?:temp(?:erature)?|column\s*temperature|evaporator\s*temperature|T)\b\s*(?:[:=]|of|at|was|is)?\s*([+-]?\d+(?:\.\d+)?)\s*(C|F|K)\b",
    re.I,
)
RE_FLOW = re.compile(
    r"\bflow[- ]*rate\b\s*(?:[:=]|of|at|was|is|were)?\s*([0-9]+(?:\.\d+)?)\s*(mL|ml|uL)\s*/\s*(min|minute|min\.)\b",
    re.I,
)
RE_CONC = re.compile(
    r"\b(?:sample\s+concentrations?|injected\s*polymer\s*conc(?:entration)?|injection\s*conc(?:entration)?|conc(?:entration)?)\b\s*(?:[:=]|of|were|was)?\s*([0-9]+(?:\.\d+)?(?:\s*(?:-|to)\s*[0-9]+(?:\.\d+)?)?)\s*(g/?L|g\s*/\s*L|mg/?mL|mg\s*/\s*mL|wt%|vol%)\b",
    re.I,
)
RE_INJECT_SOL = re.compile(r"\b(?:injected\s*polymer\s*(?:solvent|solution)|injection\s*solvent|dissolved\s*in)\s*[:=]?\s*([A-Za-z0-9\- /,\.]+?)(?=;|,|\.|\)|\n)", re.I)
RE_INJECT_VOL = re.compile(r"\b([0-9]+(?:\.\d+)?)\s*(uL|mL)\s*(?:was\s*)?(?:injected|injection)\b", re.I)

RE_PORE = re.compile(r"\b(?:pore\s*size|pore)\s*[:=Z]?\s*([0-9]+(?:\.\d+)?)\s*(A|nm|um)\b", re.I)
RE_PARTICLE = re.compile(r"\b(?:particle\s*(?:size|diameter))\s*[:=Z]?\s*([0-9]+(?:\.\d+)?)\s*(um|nm|mm)?\b", re.I)
RE_DIM = re.compile(r"\b([0-9]{2,4})\s*[x]\s*([0-9]+(?:\.\d+)?)\s*(mm|m[m]?)\b", re.I)
RE_NUM_COLS = re.compile(r"\b(?:connected\s+in\s+series|x)\s*([0-9]+)\s*(?:columns?|cols?)\b", re.I)
RE_MANUF_LINE = re.compile(r"\b(?:Agilent|Waters|Jordi|Tosoh|Phenomenex|Shimadzu|Thermo(?:\s*Fisher)?|Supelco|Macherey-Nagel|Chromtech)\b.*?(?:,\s*[A-Za-z].*?)?(?=;|\.|\)|\n)", re.I)

RE_COLUMN_SENT = re.compile(r"(?:\bcolumn\b|col\.)[^\.:\n]*", re.I)
RE_SOLVENT_MIX = re.compile(r"\b(?:mobile\s*phase|eluent|solvent\s*mixture)\b\s*[:=]?\s*([A-Za-z0-9\- /]+?)(?:\s*,?\s*(\d+(?:\.\d+)?)\s*(wt%|vol%|%|v/v|w/w))?(?=;|\.|\)|\n)", re.I)
RE_SOLVENT_PAREN = re.compile(r"\b([A-Za-z][A-Za-z0-9\-/ ]+?)\s*\((\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?)\s*(wt%|vol%|%|v/v|w/w)\)")

RE_SOLVENT_COMPONENT = re.compile(
    r"\b(\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?)\s*(wt%|w/w|vol%|v/v|%)\s*([A-Za-z][A-Za-z0-9\-]+)\b",
    re.I,
)
RE_SOLVENT_COMPONENT_PAREN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*%\s*\((w/w|v/v)\)\s*([A-Za-z][A-Za-z0-9\-]+)\b",
    re.I,
)
RE_SOLVENT_MIXTURE = re.compile(r"\b([A-Za-z][A-Za-z0-9]+(?:\s*-\s*[A-Za-z][A-Za-z0-9]+)+)\b")
RE_SOLVENT_MIXTURE_SLASH = re.compile(r"\b([A-Za-z][A-Za-z0-9\-]+(?:\s*/\s*[A-Za-z][A-Za-z0-9\-]+)+)\b")
RE_SOLVENT_MIXTURE_OF = re.compile(
    r"\bmixture\s+of\s+([A-Za-z][A-Za-z0-9\-]+)\s+(?:and|with)\s+([A-Za-z][A-Za-z0-9\-]+)\b",
    re.I,
)

# -----------------------
# Singleton Classifier
# -----------------------
_CLASSIFIER_INSTANCE = None

def get_classifier() -> "ParagraphClassifier":
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        use_semantic = os.getenv("LCCC_USE_SEMANTIC", "0").strip().lower() in ("1", "true", "yes")
        _CLASSIFIER_INSTANCE = ParagraphClassifier(use_semantic=use_semantic)
    return _CLASSIFIER_INSTANCE

class ParagraphClassifier:
    def __init__(self, use_semantic: bool = False) -> None:
        self.model = None
        if use_semantic and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(_MODEL_NAME)
                logger.info(f"Loaded sentence-transformers: {_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Semantic model load failed, fallback to heuristics: {e}")

    def is_lccc(self, text: str) -> bool:
        """Determine if paragraph describes LCCC experiment"""
        t = text.lower()
        
        if any(k.lower() in t for k in _NON_LCCC_EXCLUDE):
            if not any(k.lower() in t for k in _LCCC_POS_CUES):
                return False

        # Common abbreviation in polymer chromatography literature
        if re.search(r"\bcap\b", t) and ("critical" in t or "adsorption" in t or "column" in t or "chromatograph" in t):
            return True
        
        if any(k.lower() in t for k in _LCCC_POS_CUES):
            return True
        
        if "critical" in t and "condition" in t and ("chromatograph" in t or "column" in t or "eluent" in t):
            return True
        
        return False

    def classify(self, para: str) -> str:
        """Classify paragraph as methods, results, or other"""
        p = para.lower()
        method_cues = ["experimental", "materials", "methods", "procedure", "chromatograph", "column", "eluent", "mobile phase", "stationary phase", "flow rate", "injection", "detector", "temperature"]
        result_cues = ["results", "we observed", "we found", "figure", "table", "elution", "peak", "chromatogram", "distribution", "resolution"]
        
        score_methods = sum(p.count(k) for k in method_cues)
        score_results = sum(p.count(k) for k in result_cues)
        
        if self.is_lccc(para):
            score_methods += 3
        
        if score_methods >= max(1, score_results):
            return "methods_results"
        if score_results > 0:
            return "results"
        return "other"

# -----------------------
# Text Loading and Cleaning
# -----------------------
def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input text not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(raw: str) -> List[str]:
    text = re.sub(r"\[[0-9]{1,3}\]", " ", raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Split on blank lines OR on likely sentence boundaries created by PDF-to-text line wrapping.
    paras = re.split(r"\n\s*\n|(?<=[.!?])\s*\n+(?=[A-Z])", text)
    return [p.strip() for p in paras if p and len(p.strip()) > 20]

def classify_paragraphs(paragraphs: List[str]) -> List[Tuple[str, str]]:
    clf = get_classifier()
    return [(clf.classify(p), p) for p in paragraphs]

def extract_global_context(paragraphs: List[str]) -> Tuple[List[StationaryPhase], List[SolventDetail], TechnicalDetail]:
    """
    Extract document-level chromatography context (columns, solvents, technical settings).
    This helps attach method conditions to result paragraphs that only mention polymers.
    """
    cols: List[StationaryPhase] = []
    sols: List[SolventDetail] = []
    tech = TechnicalDetail()
    clf = get_classifier()

    # Try to restrict "global" context to the methods section when headings exist.
    start_idx = None
    end_idx = None
    for i, p in enumerate(paragraphs):
        if re.match(r"^\s*(experimental|materials\s+and\s+methods|methods)\b", p, re.I):
            start_idx = i
            break
    if start_idx is not None:
        for i in range(start_idx + 1, len(paragraphs)):
            if re.match(r"^\s*(results|results\s+and\s+discussion|discussion|conclusion)\b", paragraphs[i], re.I):
                end_idx = i
                break

    if start_idx is None:
        selected = paragraphs[: min(25, len(paragraphs))]
    else:
        selected = paragraphs[start_idx : (end_idx if end_idx is not None else len(paragraphs))]

    for para in selected:
        t = normalize_for_parsing(para)
        t_l = t.lower()

        # Avoid pulling SEC/GPC context into LCCC conditions
        if re.search(r"\b(?:sec|gpc)\b|size exclusion|gel permeation", t_l):
            # Some LCCC paragraphs mention "SEC behavior" as a comparison; keep those.
            if not clf.is_lccc(para):
                continue

        if re.search(r"\bfollowing\s+columns\b|\bcolumns\s+were\s+used\b", t_l):
            cols.extend(_extract_stationary_phases(t))
        elif "column" in t_l and (RE_DIM.search(t) or RE_PORE.search(t) or RE_PARTICLE.search(t) or RE_MANUF_LINE.search(t) or "columns" in t_l):
            # Strong evidence of an actual column spec in the methods section.
            cols.extend(_extract_stationary_phases(t))

        sols.extend(_extract_solvents(t))

        # Technical settings vary across techniques; only pull document-level tech from LCCC context
        # when it's explicitly an LCCC paragraph (prevents SEC conditions from becoming defaults).
        if clf.is_lccc(para):
            tech = _merge_technical_fill(tech, _extract_technical(t))

    # Deduplicate solvents across paragraphs
    dedup_s: Dict[Tuple[str, str, str], SolventDetail] = {}
    for s in sols:
        key = (s.solvent or "", s.ratio or "", s.ratio_units or "")
        dedup_s[key] = s
    sols = list(dedup_s.values())

    cols = _dedup_stationary_phases(cols)
    return cols, sols, tech

# -----------------------
# Context-Aware Polymer Extraction
# -----------------------
def _build_paragraph_positions(text: str, paragraphs: List[str]) -> List[Tuple[int, int, int]]:
    positions: List[Tuple[int, int, int]] = []
    cursor = 0
    for idx, para in enumerate(paragraphs):
        found = text.find(para, cursor)
        if found == -1:
            compact_para = re.sub(r"\s+", " ", para.strip())
            compact_text = re.sub(r"\s+", " ", text[cursor:])
            match = re.search(re.escape(compact_para), compact_text)
            if match:
                start = cursor + match.start()
                end = start + len(para)
                positions.append((start, end, idx))
                cursor = end
                continue
            positions.append((cursor, cursor + len(para), idx))
            cursor += len(para)
            continue
        start = found
        end = found + len(para)
        positions.append((start, end, idx))
        cursor = end
    return positions

def _build_alias_map(known_polymers: Set[str]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {p: [p] for p in known_polymers}
    for abbr, entry in polymer_abbrev_dict.items():
        canonical = normalize_name(entry["name"])
        if canonical in aliases:
            aliases[canonical].append(abbr)
    return aliases

def _alias_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias)
    if re.fullmatch(r"[A-Z0-9\-]{2,10}", alias):
        return re.compile(rf"\b{escaped}\b", re.I)
    return re.compile(rf"(?<!\w){escaped}(?:s)?(?!\w)", re.I)

def find_polymer_mentions(text: str, paragraphs: List[str], known_polymers: Set[str]) -> List[PolymerMention]:
    """
    Find all mentions of known polymers in text with context windows.
    Returns list of PolymerMention objects with surrounding context.
    """
    mentions: List[PolymerMention] = []
    para_positions = _build_paragraph_positions(text, paragraphs)
    alias_map = _build_alias_map(known_polymers)
    seen_mentions: Set[Tuple[int, int, str]] = set()

    for canonical, aliases in alias_map.items():
        for alias in aliases:
            pattern = _alias_pattern(alias)
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                key = (start_pos, end_pos, canonical)
                if key in seen_mentions:
                    continue
                seen_mentions.add(key)

                para_idx = None
                for p_start, p_end, idx in para_positions:
                    if p_start <= start_pos < p_end:
                        para_idx = idx
                        break

                if para_idx is None:
                    continue

                context_start = max(0, start_pos - 200)
                context_end = min(len(text), end_pos + 200)

                context_before = text[context_start:start_pos].strip()
                context_after = text[end_pos:context_end].strip()
                full_context = paragraphs[para_idx]

                mention = PolymerMention(
                    polymer_name=match.group(0),
                    canonical_name=canonical,
                    position=start_pos,
                    context_before=context_before[-150:] if len(context_before) > 150 else context_before,
                    context_after=context_after[:150] if len(context_after) > 150 else context_after,
                    full_context=full_context,
                    para_idx=para_idx,
                )
                mentions.append(mention)

    logger.info(f"Found {len(mentions)} polymer mentions in text")
    return mentions

# -----------------------
# Extraction helpers
# -----------------------
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

def normalize_for_parsing(text: str) -> str:
    """
    Normalize common PDF/OCR artifacts so regex-based extraction is more robust.
    This function is intentionally lightweight (no external deps).
    """
    if not text:
        return ""

    t = text
    # Normalize common PDF artifact where "x" (multiplication sign) becomes stray U+00C2 between numbers.
    t = re.sub(r"(\d)\s*\u00c2\s*(\d)", r"\1x\2", t)

    # Common unit / glyph normalizations
    t = t.replace("\u03bc", "u")   # greek mu used for micro
    t = t.replace("\u00b5", "u")   # micro sign
    t = t.replace("\u00d7", "x")   # multiplication sign
    t = t.replace("\u00b0", "")    # degree sign
    t = t.replace("\u00ba", "")    # masculine ordinal often used as degree

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

    # Common OCR: "25.0 8C" used as "25.0 degC"
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s*8\s*([CFK])\b", r"\1 \2", t)

    # Replace remaining non-ascii with spaces to keep word boundaries sane
    t = _NON_ASCII_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _normalize_value(val: Optional[str]) -> Optional[str]:
    if val is None or val == "":
        return None
    return val.strip()

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
            stop = {"the", "a", "an", "system", "column", "columns", "col", "col."}
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
            # Remove leading quantities and generic descriptors when present.
            s = re.sub(r"^(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b\s+", "", s, flags=re.I)
            s = re.sub(r"^(?:reversed\s+phase|normal\s+phase)\b\s+", "", s, flags=re.I)
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
            if best.lower() not in {"rp", "rp18", "rp-18", "c18", "ods"}:
                sp.column_name = _normalize_value(best)

        m_pore = RE_PORE.search(t)
        if m_pore:
            sp.pore_size = _normalize_value(f"{m_pore.group(1)} {m_pore.group(2)}")

        m_part = RE_PARTICLE.search(t)
        if m_part:
            unit = m_part.group(2) or ""
            if unit.lower() == "mm":
                try:
                    if float(m_part.group(1)) <= 50:
                        unit = "um"
                except Exception:
                    pass
            sp.particle_diameter = _normalize_value(f"{m_part.group(1)} {unit}".strip())

        m_dim = RE_DIM.search(t)
        if m_dim:
            sp.column_dimensions = _normalize_value(f"{m_dim.group(1)}x{m_dim.group(2)} {m_dim.group(3)}")

        m_num = RE_NUM_COLS.search(t)
        if m_num:
            sp.number_of_columns = _normalize_value(m_num.group(1))
        else:
            # e.g. "three reversed phase ... columns"
            m_num2 = re.search(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b(?:\s+[A-Za-z0-9\-]{1,15}){0,6}\s+columns?\b", t, re.I)
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
            sp.manufacturer = _normalize_value(manuf_candidates[-1])

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

        for term in _COLUMN_PHASE_TERMS:
            if term.lower() in t.lower():
                phase = term
                break
        sp.phase = phase

        mat_m = re.search(r"\b(?:material|packed with)\s*[:=]?\s*([A-Za-z0-9% \-\(\)\/]+?)(?=;|\.|,|\)|\n)", t, re.I)
        pol_m = None
        if re.search(r"\b100%\s*poly\([^)]+\)", t, re.I):
            pol_m = re.search(r"\bpoly\([^)]+\)", t)
        elif "packed with" in t_l or "material" in t_l:
            pol_m = re.search(r"\bpoly\([^)]+\)", t)
        sp.material = _normalize_value(pol_m.group(0) if pol_m else (mat_m.group(1) if mat_m else None))

        # If we have a "columns (...)" list, append it to the column_name for traceability.
        if sp.column_name:
            m_set = re.search(r"\bcolumns?\s*\(([^)]+)\)", t, re.I)
            if m_set and re.search(r"\d+\s*-\s*\d+", m_set.group(1)):
                appendix = re.sub(r"\s+", " ", m_set.group(1)).strip()
                if len(appendix) <= 80 and appendix.lower() not in sp.column_name.lower():
                    sp.column_name = _normalize_value(f"{sp.column_name} ({appendix})")

        mod_m = re.search(r"\b(?:modified|grafted|bonded)\s*with\s*([A-Za-z0-9 \-\(\)\/]+?)(?=;|\.|,|\)|\n)", t, re.I)
        sp.modification = _normalize_value(mod_m.group(1) if mod_m else None)
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

        for m in RE_SOLVENT_MIX.finditer(t):
            solvent = re.sub(r"^(?:in|on|with|and)\s+", "", m.group(1).strip(), flags=re.I)
            if not looks_like_solvent_name(solvent):
                continue
            sols.append(SolventDetail(
                solvent=_normalize_value(solvent),
                ratio=_normalize_value(m.group(2)) if m.group(2) else None,
                ratio_units=_normalize_value(m.group(3)) if m.group(3) else None
            ))
        for m in RE_SOLVENT_PAREN.finditer(t):
            solvent = re.sub(r"^(?:in|on|with|and)\s+", "", m.group(1).strip(), flags=re.I)
            if not looks_like_solvent_name(solvent):
                continue
            sols.append(SolventDetail(
                solvent=_normalize_value(solvent),
                ratio=_normalize_value(m.group(2)),
                ratio_units=_normalize_value(m.group(3))
            ))

        # Capture mixture names that appear anywhere (used for attaching component ratios).
        mixtures: List[Tuple[int, int, str]] = []
        for m in RE_SOLVENT_MIXTURE.finditer(t):
            mix = m.group(1).replace(" ", "")
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
            sols.append(SolventDetail(solvent=_normalize_value(mix), ratio=None, ratio_units=None))

        # Extract ratio lists like "butanone/cyclohexane: 70:30; 92:8; 97:3"
        for m_start, m_end, mix in mixtures:
            tail = t[m_end : m_end + 220]
            for rm in re.finditer(r"\b(\d+(?:\.\d+)?)([:/])(\d+(?:\.\d+)?)(?:\s*(vol%|wt%|v/v|w/w|%))?", tail, re.I):
                a = rm.group(1)
                delim = rm.group(2)
                b = rm.group(3)
                units = rm.group(4)
                ratio = f"{a}{delim}{b}"
                sols.append(SolventDetail(solvent=_normalize_value(mix), ratio=_normalize_value(ratio), ratio_units=_normalize_value(units) if units else None))

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

            if mix:
                solvent = f"{mix} ({component})"
            else:
                solvent = component

            if looks_like_solvent_name(solvent.replace(f" ({component})", "")) or mix:
                sols.append(SolventDetail(
                    solvent=_normalize_value(solvent),
                    ratio=_normalize_value(ratio),
                    ratio_units=_normalize_value(units),
                ))

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

            solvent = f"{mix} ({component})" if mix else component
            if looks_like_solvent_name(solvent.replace(f" ({component})", "")) or mix:
                sols.append(SolventDetail(solvent=_normalize_value(solvent), ratio=_normalize_value(ratio), ratio_units=_normalize_value(units)))

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
            td.temperature = _normalize_value(f"{mT.group(1)} {mT.group(2)}")
        mF = RE_FLOW.search(t)
        if mF:
            td.flow_rate = _normalize_value(f"{mF.group(1)} {mF.group(2)}/{mF.group(3)}")
        mC = RE_CONC.search(t)
        if mC:
            td.injected_polymer_concentration = _normalize_value(f"{mC.group(1)} {mC.group(2)}")
        mS = RE_INJECT_SOL.search(t)
        if mS:
            td.injected_polymer_solvent_solution = _normalize_value(mS.group(1))
        mV = RE_INJECT_VOL.search(t)
        if mV:
            td.injection_volume = _normalize_value(f"{mV.group(1)} {mV.group(2)}")
    except Exception as e:
        logger.warning(f"Error extracting technical details: {e}")

    return td

# -----------------------
# Context aggregation helpers
# -----------------------
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
                sp.column_name = _normalize_value(name)

            # Parse known attributes from entry text
            m_dim = RE_DIM.search(entry)
            if m_dim:
                sp.column_dimensions = _normalize_value(f"{m_dim.group(1)}x{m_dim.group(2)} {m_dim.group(3)}")

            m_part = RE_PARTICLE.search(entry)
            if m_part:
                unit = m_part.group(2) or ""
                if unit.lower() == "mm":
                    try:
                        if float(m_part.group(1)) <= 50:
                            unit = "um"
                    except Exception:
                        pass
                sp.particle_diameter = _normalize_value(f"{m_part.group(1)} {unit}".strip())

            m_pore = RE_PORE.search(entry)
            if m_pore:
                sp.pore_size = _normalize_value(f"{m_pore.group(1)} {m_pore.group(2)}")

            pol_m = re.search(r"\bpoly\([^)]+\)", entry)
            if pol_m:
                sp.material = _normalize_value(pol_m.group(0))

            phase_m = re.search(r"\b([A-Za-z0-9\-]{2,20})\s+phase\b", entry, re.I)
            if phase_m:
                sp.phase = _normalize_value(phase_m.group(1))

            parens = re.findall(r"\(([^)]+)\)", entry)
            manuf_candidates = [p.strip() for p in parens if "," in p]
            if manuf_candidates:
                sp.manufacturer = _normalize_value(manuf_candidates[-1])

            if _has_stationary_phase_data(sp):
                phases.append(sp)
                found_list_style = True

    # Generic single-column extraction from any paragraph
    if not found_list_style:
        sp = _extract_stationary_phase(t)
        if _has_stationary_phase_data(sp):
            phases.append(sp)

    return _dedup_stationary_phases(phases)

def _merge_technical(base: TechnicalDetail, override: TechnicalDetail) -> TechnicalDetail:
    merged = TechnicalDetail(
        temperature=override.temperature or base.temperature,
        flow_rate=override.flow_rate or base.flow_rate,
        injected_polymer_concentration=override.injected_polymer_concentration or base.injected_polymer_concentration,
        injected_polymer_solvent_solution=override.injected_polymer_solvent_solution or base.injected_polymer_solvent_solution,
        injection_volume=override.injection_volume or base.injection_volume,
    )
    return merged

def _merge_technical_fill(base: TechnicalDetail, add: TechnicalDetail) -> TechnicalDetail:
    """
    Fill-only merge: keep existing values and only populate missing fields.
    Useful when building a global context where later paragraphs may contain
    other temperatures (e.g., detector evaporator) that should not override
    the main column temperature.
    """
    merged = TechnicalDetail(
        temperature=base.temperature or add.temperature,
        flow_rate=base.flow_rate or add.flow_rate,
        injected_polymer_concentration=base.injected_polymer_concentration or add.injected_polymer_concentration,
        injected_polymer_solvent_solution=base.injected_polymer_solvent_solution or add.injected_polymer_solvent_solution,
        injection_volume=base.injection_volume or add.injection_volume,
    )
    return merged

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

# -----------------------
# Context-Aware Entity Extraction
# -----------------------
def extract_entities_context_aware(
    mentions: List[PolymerMention],
    labeled_paragraphs: List[Tuple[str, str]],
    clf: "ParagraphClassifier",
    global_columns: List[StationaryPhase],
    global_solvents: List[SolventDetail],
    global_tech: TechnicalDetail,
) -> List[Dict[str, Any]]:
    """
    Extract LCCC experiments using context-aware polymer mentions.
    Groups mentions by paragraph so we don't create many duplicate experiments.
    Uses a global methods context to attach settings like temperature/flow rate.
    """
    items: List[Dict[str, Any]] = []

    # Group mentions by paragraph and polymer to reduce duplicates
    by_para: Dict[int, Dict[str, List[PolymerMention]]] = {}
    for m in mentions:
        by_para.setdefault(m.para_idx, {}).setdefault(m.canonical_name, []).append(m)

    global_solvents_with_ratio = [s for s in global_solvents if s.ratio]

    for para_idx in sorted(by_para.keys()):
        para_label, para_text = labeled_paragraphs[para_idx]

        # Build a small context window across neighboring segments because PDF-to-text
        # often breaks paragraphs into sentence-sized lines.
        ctx_parts: List[str] = []
        for j in range(max(0, para_idx - 2), min(len(labeled_paragraphs), para_idx + 3)):
            ctx_parts.append(labeled_paragraphs[j][1])
        ctx_text = " ".join(ctx_parts)

        # Keep a broad gate based on classifier + paragraph class.
        mention_ctx_lccc = any(clf.is_lccc(m.get_context_window()) for ms in by_para[para_idx].values() for m in ms)
        para_is_lccc = clf.is_lccc(para_text)
        ctx_is_lccc = clf.is_lccc(ctx_text)
        is_lccc_context = mention_ctx_lccc or para_is_lccc or ctx_is_lccc
        if not is_lccc_context:
            continue

        # If the paragraph itself isn't LCCC (and neither are the mention windows),
        # avoid "leaking" SEC/GPC methods into LCCC just because neighbors mention LCCC.
        if not (mention_ctx_lccc or para_is_lccc):
            if re.search(r"\b(?:sec|gpc)\b|size exclusion|gel permeation", ctx_text.lower()):
                continue

        local_cols = _extract_stationary_phases(ctx_text)
        local_sols = _extract_solvents(ctx_text)
        local_tech = _extract_technical(ctx_text)

        matched_cols = _match_stationary_phases(para_text, global_columns)
        if not matched_cols:
            matched_cols = _match_stationary_phases(ctx_text, global_columns)
        cols = matched_cols or local_cols

        tech = _merge_technical(global_tech, local_tech)

        sols = local_sols
        if not sols and global_solvents_with_ratio:
            sols = global_solvents_with_ratio

        has_data = bool(cols) or bool(sols) or _has_technical_data(tech)
        if not has_data:
            continue

        for canonical, ms in by_para[para_idx].items():
            unique_mentions = sorted({m.polymer_name for m in ms})
            context_window = ms[0].get_context_window()[:300]

            items.append({
                "para_idx": para_idx,
                "polymer_name": canonical,
                "polymer_mentions": unique_mentions,
                "context_window": context_window,
                "is_lccc": True,
                "stationary_phases": cols,
                "solvent_details": sols,
                "technical_details": tech,
            })

            logger.info(f"Para {para_idx}: Extracted LCCC data for {canonical}")

    logger.info(f"Extracted {len(items)} LCCC experiments from {len(mentions)} polymer mentions")
    return items

# -----------------------
# Context linking
# -----------------------
def link_context(extracted: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build structured output per polymer"""
    result: Dict[str, Dict[str, Any]] = {}
    exp_counter: Dict[str, int] = {}
    sig_to_exp: Dict[str, Dict[str, str]] = {}

    for item in extracted:
        poly = item["polymer_name"]
        cols: List[StationaryPhase] = item.get("stationary_phases", [])
        sols: List[SolventDetail] = item.get("solvent_details", [])
        tech: TechnicalDetail = item["technical_details"]

        if poly not in result:
            result[poly] = {}
            exp_counter[poly] = 0
            sig_to_exp[poly] = {}

        sp_dicts = [sp.to_dict() for sp in cols if _has_stationary_phase_data(sp)]
        sol_dicts = [s.to_dict() for s in sols]
        tech_dict = tech.to_dict() if _has_technical_data(tech) else None

        signature = json.dumps(
            {"stationary_phase": sp_dicts, "solvent_details": sol_dicts, "technical_details": tech_dict},
            sort_keys=True,
            ensure_ascii=True,
        )

        evidence = {
            "para_idx": item.get("para_idx"),
            "polymer_mentions": item.get("polymer_mentions", []),
            "context_snippet": item.get("context_window", ""),
        }

        existing_id = sig_to_exp[poly].get(signature)
        if existing_id:
            exp = result[poly][existing_id]
            exp.setdefault("evidence", []).append(evidence)
            exp.setdefault("paragraph_indices", []).append(item.get("para_idx"))

            # Merge mention strings
            merged_mentions = set(exp.get("polymer_mentions", []))
            merged_mentions.update(item.get("polymer_mentions", []))
            exp["polymer_mentions"] = sorted(merged_mentions)
            continue

        exp_counter[poly] += 1
        exp_id = f"experiment_{exp_counter[poly]}"
        sig_to_exp[poly][signature] = exp_id

        result[poly][exp_id] = {
            "polymer_mention": (item.get("polymer_mentions") or [""])[0],
            "polymer_mentions": item.get("polymer_mentions", []),
            "context_snippet": item.get("context_window", ""),
            "paragraph_indices": [item.get("para_idx")],
            "evidence": [evidence],
            "stationary_phase": sp_dicts,
            "solvent_details": sol_dicts,
            "technical_details": [tech.to_dict()] if tech_dict else [],
        }

    return result

def validate_output(data: Dict[str, Dict[str, Any]], known_polymers: Set[str]) -> bool:
    """Validate extracted data structure"""
    try:
        for poly, exps in data.items():
            if poly not in known_polymers:
                logger.warning(f"Unknown polymer in output: {poly}")
                return False
            for exp_id, exp in exps.items():
                if not isinstance(exp, dict) or not all(k in exp for k in ["stationary_phase", "solvent_details", "technical_details"]):
                    logger.warning(f"Invalid experiment structure: {exp_id}")
                    return False
        return True
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

# -----------------------
# Orchestration
# -----------------------
def run_pipeline(text: str, use_validation: bool = True, threshold_ratio: float = 0.5) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Run context-aware LCCC extraction pipeline.
    """
    start = time.time()
    logger.info("Starting context-aware LCCC extraction pipeline...")
    
    # Step 1: Extract all polymers from text
    logger.info("Step 1: Extracting polymers from text...")
    all_polymers = extract_polymers(text)
    logger.info(f"  Found {len(all_polymers)} unique polymers")
    for poly, count in list(all_polymers.items())[:5]:
        logger.info(f"    - {poly}: {count}")
    
    # Step 2: Validate polymers (optional)
    if use_validation:
        logger.info("Step 2: Validating polymers against PubChem...")
        validated_polymers = validate_polymers(all_polymers)
        logger.info(f"  Validated {len(validated_polymers)} polymers")
        if not validated_polymers and all_polymers:
            logger.warning("PubChem validation returned no polymers; falling back to unvalidated set")
            validated_polymers = all_polymers
    else:
        validated_polymers = all_polymers
        logger.info("Step 2: Skipping validation (use_validation=False)")
    
    # Step 3: Select main polymers
    logger.info(f"Step 3: Selecting main polymers (threshold={threshold_ratio})...")
    main_polymers = select_main_polymers(validated_polymers, threshold_ratio=threshold_ratio)
    logger.info(f"  Selected {len(main_polymers)} main polymers")
    for poly, count in main_polymers.items():
        logger.info(f"    - {poly}: {count}")
    
    known_polymers_set = set(main_polymers.keys())
    
    # Step 4: Clean and classify paragraphs
    logger.info("Step 4: Cleaning and classifying paragraphs...")
    paragraphs = clean_text(text)
    logger.info(f"  Split into {len(paragraphs)} paragraphs")
    
    labeled = classify_paragraphs(paragraphs)
    lccc_count = sum(1 for l, _ in labeled if l in ("methods_results", "results"))
    logger.info(f"  Classified {lccc_count} relevant paragraphs")

    # Step 4b: Extract global methods context (columns/solvents/tech settings)
    logger.info("Step 4b: Extracting global chromatography context...")
    global_cols, global_sols, global_tech = extract_global_context(paragraphs)
    logger.info(f"  Global columns: {len(global_cols)}")
    logger.info(f"  Global solvent entries: {len(global_sols)}")
    if _has_technical_data(global_tech):
        logger.info(f"  Global technical: {global_tech.to_dict()}")
    
    # Step 5: Find polymer mentions with context
    logger.info("Step 5: Finding polymer mentions with context windows...")
    mentions = find_polymer_mentions(text, paragraphs, known_polymers_set)
    logger.info(f"  Found {len(mentions)} polymer mentions")
    
    # Step 6: Extract LCCC entities using context
    logger.info("Step 6: Extracting LCCC experimental details from context...")
    clf = get_classifier()
    extracted = extract_entities_context_aware(
        mentions,
        labeled,
        clf,
        global_columns=global_cols,
        global_solvents=global_sols,
        global_tech=global_tech,
    )
    logger.info(f"  Extracted {len(extracted)} LCCC experiments")
    
    # Step 7: Link context
    logger.info("Step 7: Linking context and building experiments...")
    linked = link_context(extracted)
    logger.info(f"  Linked into {len(linked)} polymers")
    
    # Step 8: Validate
    if not validate_output(linked, known_polymers_set):
        logger.warning("Output validation failed")
    
    elapsed = time.time() - start
    logger.info(f"Pipeline completed in {elapsed:.2f}s")
    
    metadata = {
        "total_polymers_found": len(all_polymers),
        "validated_polymers": len(validated_polymers),
        "main_polymers": len(main_polymers),
        "total_paragraphs": len(paragraphs),
        "lccc_paragraphs": lccc_count,
        "polymer_mentions_found": len(mentions),
        "extracted_experiments": len(extracted),
        "processing_time_seconds": elapsed,
        "main_polymers_list": list(main_polymers.keys())
    }
    
    return linked, metadata

def save_json(data: Dict[str, Dict[str, Any]], out_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save structured JSON to file with optional metadata"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    output = {
        "metadata": metadata or {},
        "experiments": data
    }
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f" Saved structured JSON to {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract LCCC experiments with context-aware polymer detection")
    ap.add_argument("input", help="Path to input text file")
    ap.add_argument("-o", "--output", default="extracted_lccc_data.json", help="Output JSON path")
    ap.add_argument("--no-validation", action="store_true", help="Skip PubChem validation")
    ap.add_argument("--threshold", type=float, default=0.5, help="Polymer selection threshold (0.0-1.0)")
    args = ap.parse_args()
    
    raw = load_text(args.input)
    result, metadata = run_pipeline(raw, use_validation=not args.no_validation, threshold_ratio=args.threshold)
    save_json(result, args.output, metadata)
