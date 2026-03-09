from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

from ..logging_config import logger
from ..models import SolventDetail, StationaryPhase, TechnicalDetail
from ..text.match import value_in_text
from ..text.normalization import normalize_for_parsing
from .client import LLMClient, LLMConfig


# ─────────────────────────────────────────────────────────────────────────────
# Field definitions
# ─────────────────────────────────────────────────────────────────────────────
LCCC_FIELDS = ["critical_polymer_unit", "polymer_system", "SP_FIELDS", "SOL_FIELDS", "TECH_FIELDS", "separation_behavior"]

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

# ─────────────────────────────────────────────────────────────────────────────
# Shared prompt constants
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = (
    '{"LCCC_conditions":[{'
    '"critical_polymer_unit":null,'
    '"polymer_system":null,'
    '"SP_FIELDS":{'
    '"column_name":null,"material":null,"modification":null,"pore_size":null,'
    '"particle_diameter":null,"column_dimensions":null,"number_of_columns":null,'
    '"manufacturer":null,"phase":null},'
    '"SOL_FIELDS":{"solvent":null,"ratio":null,"ratio_units":null},'
    '"TECH_FIELDS":{'
    '"temperature":null,"flow_rate":null,"injected_polymer_concentration":null,'
    '"injected_polymer_solvent_solution":null,"injection_volume":null},'
    '"separation_behavior":{'
    '"critical_block_behavior":null,"non_critical_block_behavior":null,"purpose":null}'
    '}]}'
)

_COLUMN_SCHEMA = (
    '{"columns":[{'
    '"column_name":null,"material":null,"modification":null,"pore_size":null,'
    '"particle_diameter":null,"column_dimensions":null,"number_of_columns":null,'
    '"manufacturer":null,"phase":null'
    '}]}'
)

# ─────────────────────────────────────────────────────────────────────────────
# Precision-engineered field guide
# Every rule is placed in a SHORT, NUMBERED, scannable form so the LLM cannot
# skip past it.  Critical rules are marked ⚠ and put FIRST within each field.
# ─────────────────────────────────────────────────────────────────────────────
_FIELD_GUIDE = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD EXTRACTION RULES  (read every rule before writing any value)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[critical_polymer_unit]
  The specific polymer block whose retention is independent of molar mass.
  ✓ Extract the abbreviation or name exactly as written: "PEO" "PS" "PMMA" "PEG"
  ✗ "polymer block"  "the critical unit"  (too vague — null these)
  ⚠ If not explicitly stated: null. Do NOT infer from context.

[polymer_system]
  The full copolymer architecture being analyzed (not just the critical unit).
  ✓ "PEO-b-PLLA diblock"  "PLLA-b-PEO-b-PLLA triblock"  "PS-b-PMMA diblock"
  ✗ "PEO"  (that is the critical unit)  "block copolymer"  (too vague — null)
  ⚠ Diblock vs triblock = DIFFERENT polymer_system values even if column is the same.

[SP_FIELDS — stationary phase / column]

  column_name:
    Full brand + model name as printed in the text.
    ✓ "Luna C18"  "Nucleosil 100 C18"  "Jordi Gel DVB"  "Discovery HS-PEG"
    ✗ "C18 column"  "silica column"  (generic descriptions — null these)

  material:
    ✓ "silica"  "divinylbenzene"  "cross-linked polystyrene"
    ✗ "stationary phase"  (too vague — null)

  modification:
    Surface chemistry / bonded phase label ONLY.
    ✓ "C18"  "diol"  "PEG"  "amino"  "C8"  "OH"
    ⚠ "reversed phase" belongs in PHASE, not here.

  pore_size:
    ⚠ ALWAYS include the unit token.
    ✓ "100 A"  "300 A"  "500 A"  "100 Angstrom"
    ✗ "100"  (no unit — null this)

  particle_diameter:
    ⚠ ALWAYS include the unit token.
    ✓ "5 um"  "3 um"  "10 um"  "5 micron"
    ✗ "5"  (no unit — null this)

  column_dimensions:
    ⚠ Must include BOTH length AND diameter with units.
    ✓ "250 x 4.6 mm"  "300 x 7.8 mm"  "150 x 4.0 mm"
    ✗ "250 mm"  (only one dimension — null this)

  number_of_columns:
    Integer. ONLY populate when the text EXPLICITLY states multiple identical
    columns connected in series. Otherwise null.

  manufacturer:
    ✓ "Phenomenex"  "Waters"  "Macherey-Nagel"  "Supelco"  "Polymer Labs"
    ✗ "the manufacturer"  (null)

  phase:
    Separation MODE only.
    ✓ "reversed phase"  "normal phase"  "size exclusion"  "HILIC"
    ⚠ "C18" goes in modification, NOT here.

[SOL_FIELDS — mobile phase]

  solvent:
    Component names separated by " / " for mixtures, capitalized as written.
    ✓ "acetonitrile / water"  "THF"  "Water / Acetonitrile"

  ratio:
    ⚠ ALWAYS include BOTH parts of the ratio. A one-sided ratio is wrong.
    ✓ "60/40"  "54:46"  "56/44"  "70/30"
    ✗ "60"  "54"  (incomplete — null these)

  ratio_units:
    ✓ "v/v"  "vol%"  "wt%"  "w/w"

[TECH_FIELDS]

  temperature:
    ⚠⚠ CRITICAL RULE — read this carefully:
    • Strip the degree/unit symbol: "35 degC" → "35"   "54 C" → "54"
    • A RANGE means the critical temperature was NOT found → null.
      Examples that MUST become null:
        "6-54 C"  "between 6 and 54 C"  "6 to 54 C"  "20–40 C"
    • A single value is correct → keep it.
    • ⚠ ISOLATION: If the text names multiple temperatures for DIFFERENT
      conditions, assign only the temperature that belongs to THIS specific
      mobile phase. Never borrow a temperature from another condition.

  flow_rate:
    Strip units. ✓ "1.0"  "0.5"  "2"

  injected_polymer_concentration:
    Keep units. ✓ "2 mg/mL"  "1 g/L"  "0.5 wt%"

  injected_polymer_solvent_solution:
    ✓ "THF"  "chloroform"  "mobile phase"

  injection_volume:
    Strip units. ✓ "20"  "50"  "100"

[separation_behavior]

  critical_block_behavior:
    ⚠ Must be a COMPLETE sentence: [BLOCK NAME] + [VERB] + [BEHAVIOR].
    ✓ "PEO block elutes at the same retention volume regardless of chain length"
    ✓ "retention of PEO block is independent of its molecular weight"
    ✗ "PEO block"            (no behavior — null this)
    ✗ "chromatographically invisible"  (no subject — null this)

  non_critical_block_behavior:
    ⚠ Must be a COMPLETE sentence naming the non-critical block.
    ✓ "PLLA block elutes in order of its molecular weight"
    ✓ "PEO-b-PLLA copolymer separates identically to PLLA homopolymer"
    ✗ "PLLA"  (not a sentence — null this)

  purpose:
    One clear sentence explaining WHY LCCC is used here.
    ✓ "to determine the molar mass distribution of the PLLA block independently"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

# ─────────────────────────────────────────────────────────────────────────────
# Accuracy contract — tightened with explicit failure modes
# ─────────────────────────────────────────────────────────────────────────────
_ANTI_HALLUCINATION = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACCURACY CONTRACT  (violations produce unusable output)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rule 1 — VERBATIM SOURCE
  Every value must come DIRECTLY from the text.
  Allowed: normalize capitalization; strip units from numeric fields.
  Forbidden: paraphrase, infer, compute, or combine values across sentences.

Rule 2 — NULL IS THE CORRECT DEFAULT
  If a field value is not explicitly stated → null.
  An incomplete but accurate record is far more valuable than a complete
  record with invented values. DO NOT guess.

Rule 3 — NO CROSS-CONTAMINATION BETWEEN CONDITIONS
  Never copy a value from one experimental condition into another.
  This is especially critical for temperature and solvent ratio:
  if condition A uses 35 C and condition B uses 54 C, do not assign
  35 to condition B just because the text mentions both.

Rule 4 — SCHEMA NULLS ARE PLACEHOLDERS
  The null values in the schema above are placeholders.
  Never copy any value shown in the schema into your answer.

Rule 5 — NO UNIT INFERENCE
  If a numeric value appears WITHOUT a unit in the text, do not add a unit.
  If a unit cannot be extracted verbatim → null the field or omit the unit.
  Exception: temperature always has its unit stripped per the field guide.

Rule 6 — MANDATORY SELF-AUDIT  (execute before writing your answer)
  For EVERY non-null value you intend to write, locate the exact phrase
  in the text that justifies it. Then check:
  (a) Temperature: is it a single number? If a range → null.
  (b) Ratio: does it contain BOTH parts (e.g. "60/40")? If one part → null.
  (c) critical_block_behavior: does it contain a subject + verb + behavior?
      If missing any of these → null.
  (d) non_critical_block_behavior: same check.
  (e) pore_size / particle_diameter: does it include the unit token? If not → null.
  (f) column_dimensions: does it include both dimensions? If not → null.
  If a check fails → null that field. Do not adjust the value to make it pass.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

_OUTPUT_FORMAT = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Return ONLY valid JSON. No prose, no markdown, no code fences.
• Top-level key must be "LCCC_conditions" with an array value.
• If nothing qualifies → {"LCCC_conditions":[]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LLMChunk:
    chunk_id: int
    para_start: int
    para_end: int
    raw_text: str
    llm_text: str


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────
class LLMExtractor:
    WINDOW_PARAGRAPHS = 2
    OVERLAP_PARAGRAPHS = 1
    EVIDENCE_WINDOW = 0

    SOLVENT_KEYWORDS: List[str] = [
        "acetone", "water", "methanol", "ethanol", "propanol", "isopropanol",
        "acetonitrile", "butanone", "methyl ethyl ketone", "cyclohexane",
        "hexane", "heptane", "toluene", "chloroform", "dichloromethane",
        "thf", "tetrahydrofuran", "dmf", "dmso", "buffer", "phosphate",
    ]

    KEYWORDS: List[str] = [
        "column", "columns", "cap", "critical", "critical conditions",
        "critical adsorption", "lccc", "mobile phase", "eluent",
        "wt%", "w/w", "v/v", "vol%", "flow rate", "temperature", "injection",
        "chromatographically invisible",
    ]

    CAP_KEYWORDS: List[str] = [
        "cap", "critical", "critical conditions", "critical adsorption",
        "critical point", "lccc", "chromatographically invisible",
        "adsorption-desorption", "critical elution",
    ]

    TECH_KEYWORDS: List[str] = [
        "temperature", "flow rate", "flow-rate", "injection", "loop",
        "ml/min", "mL/min", "autosampler",
    ]

    SECTION_HINTS: List[str] = [
        "experimental", "materials", "methods", "results", "discussion",
        "conclusions",
    ]

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.client = LLMClient(config)
        self.config = self.client.config
        self.available = self.client.available
        self.models = self.client.models
        self.consensus_min = max(1, int(self.config.consensus_min or 1))
        self.high_accuracy = (
            os.getenv("LCCC_LLM_HIGH_ACCURACY", "1").strip().lower() in ("1", "true", "yes")
        )
        self.max_global_chars = int(os.getenv("LCCC_LLM_GLOBAL_CHARS", "12000"))
        self.refine_enabled = (
            os.getenv("LCCC_LLM_REFINE", "1").strip().lower() in ("1", "true", "yes")
        )
        self.refine_limit = max(1, int(os.getenv("LCCC_LLM_REFINE_LIMIT", "6")))
        self.debug = (
            os.getenv("LCCC_LLM_DEBUG", "0").strip().lower() in ("1", "true", "yes")
        )

    # ── text helpers ──────────────────────────────────────────────────────────

    def _sanitize_for_llm(self, text: str) -> str:
        if not text:
            return ""
        t = normalize_for_parsing(text)
        t = re.sub(r"\[(?:\d+|[,\-\s]){1,20}\]", " ", t)
        t = re.sub(r"\b(?:fig(?:ure)?|table)\s*\d+[a-z]?\b", " ", t, flags=re.I)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _numbers_present(self, value: str, text: str) -> bool:
        nums = re.findall(r"\d+(?:\.\d+)?", value or "")
        if not nums:
            return False
        t = normalize_for_parsing(text)
        return all(n in t for n in nums)

    def _sanitize_temperature(self, value: Any) -> Optional[str]:
        """
        Accept a single numeric temperature string; reject ranges.
        "6-37", "6-54", "20 to 40" -> None
        "35", "54", "22.5" -> "35", "54", "22.5"
        """
        if value is None:
            return None
        s = str(value).strip()
        # Reject range patterns: dash/en-dash/em-dash/tilde between digits
        if re.search(r"\d\s*[-\u2013\u2014~]\s*\d", s):
            return None
        # Reject "X to Y" patterns
        if re.search(r"\d\s+to\s+\d", s, re.I):
            return None
        # Reject "between X and Y" if accidentally included
        if re.search(r"between", s, re.I):
            return None
        nums = re.findall(r"\d+(?:\.\d+)?", s)
        if len(nums) == 1:
            return nums[0]
        return None

    def _is_relevant(self, text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in self.KEYWORDS)

    def _score_paragraph(self, text: str) -> int:
        if not text:
            return 0
        t = normalize_for_parsing(text).lower()
        score = 0
        if any(k in t for k in self.SECTION_HINTS):
            score += 1
        if any(k in t for k in self.CAP_KEYWORDS):
            score += 3
        if any(k in t for k in self.SOLVENT_KEYWORDS):
            score += 2
        if "column" in t:
            score += 2
        if re.search(r"\b\d+(?:\.\d+)?\s*(wt%|w/w|v/v|vol%|%)\b", t):
            score += 2
        if re.search(r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?", t):
            score += 1
        if re.search(r"\b\d+(?:\.\d+)?\s*(ml|min|mL|uL|ul)\b", t):
            score += 1
        if re.search(r"\b\d+(?:\.\d+)?\s*[cC]\b", t):
            score += 1
        return score

    # ── chunking ──────────────────────────────────────────────────────────────

    def _chunk_window(
        self,
        paragraphs: List[str],
        start: int,
        end: int,
        max_chars: int,
        overlap_paragraphs: int,
    ) -> List[Tuple[int, int, str]]:
        chunks: List[Tuple[int, int, str]] = []
        idx = start
        total = len(paragraphs)
        while idx <= end and idx < total:
            chunk_start = idx
            chunk_end = idx
            length = 0
            truncated_text = None
            while chunk_end <= end and chunk_end < total:
                para = paragraphs[chunk_end]
                para_len = len(para) + 2
                if length + para_len > max_chars and chunk_end > chunk_start:
                    break
                if length + para_len > max_chars and chunk_end == chunk_start:
                    truncated_text = para[:max_chars]
                    chunk_end = chunk_start + 1
                    break
                length += para_len
                chunk_end += 1
            if chunk_end <= chunk_start:
                chunk_end = chunk_start + 1
            text = (
                truncated_text if truncated_text is not None
                else "\n\n".join(paragraphs[chunk_start:chunk_end])
            )
            chunks.append((chunk_start, chunk_end - 1, text))
            if chunk_end > end:
                break
            next_start = max(chunk_end - overlap_paragraphs, chunk_start + 1)
            idx = next_start
        return chunks

    def build_chunks(self, paragraphs: List[str]) -> List[LLMChunk]:
        if not paragraphs:
            return []
        anchors = [i for i, p in enumerate(paragraphs) if self._is_relevant(p)]
        if not anchors:
            anchors = list(range(len(paragraphs)))

        windows: List[Tuple[int, int]] = []
        for idx in anchors:
            start = max(0, idx - self.WINDOW_PARAGRAPHS)
            end = min(len(paragraphs) - 1, idx + self.WINDOW_PARAGRAPHS)
            windows.append((start, end))
        windows.sort()
        merged: List[Tuple[int, int]] = []
        for start, end in windows:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))

        max_chars = max(400, int(self.config.max_context_chars))
        chunks: List[LLMChunk] = []
        chunk_id = 0
        for start, end in merged:
            for c_start, c_end, text in self._chunk_window(
                paragraphs, start, end,
                max_chars=max_chars,
                overlap_paragraphs=self.OVERLAP_PARAGRAPHS,
            ):
                llm_text = self._sanitize_for_llm(text)
                if not llm_text or not self._is_relevant(llm_text):
                    continue
                chunk_id += 1
                chunks.append(LLMChunk(
                    chunk_id=chunk_id,
                    para_start=c_start,
                    para_end=c_end,
                    raw_text=text,
                    llm_text=llm_text,
                ))
        return chunks

    # ── column catalog ────────────────────────────────────────────────────────

    def _extract_columns_catalog(self, paragraphs: List[str], full_text: str) -> List[Dict[str, Any]]:
        if not paragraphs:
            return []

        col_paras: List[str] = []
        for i, p in enumerate(paragraphs):
            if re.search(r"\bfollowing\s+columns?\b|\bcolumns?\s+were\s+used\b", p, re.I):
                col_paras.append(p)
                if i + 1 < len(paragraphs):
                    col_paras.append(paragraphs[i + 1])
        if not col_paras:
            col_paras = [p for p in paragraphs if "column" in (p or "").lower()]
        if not col_paras:
            return []

        text = self._sanitize_for_llm("\n\n".join(col_paras))
        if len(text) > 5000:
            text = text[:5000]

        prompt = f"""\
You are a scientific data extraction assistant specialised in chromatography.

TASK
----
Extract every chromatography column specification mentioned in the text.
Return ONLY valid JSON with top-level key "columns".

OUTPUT SCHEMA
-------------
{_COLUMN_SCHEMA}

FIELD RULES
-----------
column_name
  ⚠ Brand + model name EXACTLY as written. No paraphrasing.
  ✓ "Luna C18(2)"  "Nucleosil 100-5 C18"  "Jordi Gel DVB 500"
  ✗ "C18 column"  "reversed-phase column"  (too generic — null these)

manufacturer
  ✓ "Phenomenex"  "Waters"  "Macherey-Nagel"  "Supelco"  "Polymer Labs"

material
  ✓ "silica"  "divinylbenzene"  "polystyrene"
  ✗ "stationary phase"  (null)

modification
  Bonded phase / surface chemistry label ONLY.
  ✓ "C18"  "diol"  "PEG"  "amino"  "C8"
  ⚠ "reversed phase" belongs in phase, NOT here.

phase
  Separation mode only.
  ✓ "reversed phase"  "normal phase"  "size exclusion"  "HILIC"

column_dimensions
  ⚠ Must include BOTH dimensions with units.
  ✓ "250 x 4.6 mm"  "300 x 7.8 mm"
  ✗ "250 mm"  (one dimension only — null)

particle_diameter
  ⚠ Include the unit.  ✓ "5 um"  "3 um"   ✗ "5"  (null)

pore_size
  ⚠ Include the unit.  ✓ "100 A"  "300 A"   ✗ "100"  (null)

number_of_columns
  Integer only when the text EXPLICITLY states multiple identical columns
  connected in series. Otherwise null.

{_ANTI_HALLUCINATION}

EXTRACTION RULES
----------------
• One JSON object per distinct column.
• Extract EVERY column mentioned, even if only partially described.
• null for any field not explicitly stated in the text.
• Ignore non-column hardware (detectors, pumps, injectors, autosamplers).

{_OUTPUT_FORMAT}

TEXT
----
{text}
"""
        outputs = self.client.run_json(prompt)
        if not outputs:
            return []

        def _col_key(col: Dict[str, Any]) -> str:
            name = normalize_for_parsing(str(col.get("column_name") or "")).lower()
            if name:
                return f"name:{name}"
            return "|".join(
                f"{k}:{normalize_for_parsing(str(col.get(k) or '')).lower()}"
                for k in ("manufacturer", "column_dimensions", "phase", "material")
            )

        def _score(col: Dict[str, Any]) -> int:
            return sum(1 for v in col.values() if v is not None)

        per_model: List[Tuple[str, List[Dict[str, Any]]]] = []
        for model_name, data in outputs:
            cols: List[Any] = []
            if isinstance(data, dict):
                cols = data.get("columns") or []
            elif isinstance(data, list):
                cols = data
            if not isinstance(cols, list):
                continue
            cleaned = []
            for col in cols:
                if not isinstance(col, dict):
                    continue
                col_clean = {f: col.get(f) for f in SP_FIELDS}
                col_clean = self._filter_fields(
                    col_clean, SP_FIELDS, full_text,
                    soft_fields={"column_name", "material", "modification", "manufacturer", "phase"},
                )
                if any(v is not None for v in col_clean.values()):
                    cleaned.append(col_clean)
            if cleaned:
                per_model.append((model_name, cleaned))

        if not per_model:
            return []
        if len(per_model) == 1 or self.consensus_min <= 1:
            return per_model[0][1]

        support: Dict[str, set] = {}
        candidates: Dict[str, List[Dict[str, Any]]] = {}
        for model_name, cols in per_model:
            for col in cols:
                key = _col_key(col)
                support.setdefault(key, set()).add(model_name)
                candidates.setdefault(key, []).append(col)

        merged_cols: List[Dict[str, Any]] = []
        for key, cols in candidates.items():
            if len(support.get(key, set())) < self.consensus_min:
                continue
            merged_cols.append(max(cols, key=_score))
        return merged_cols

    # ── column alias helpers ──────────────────────────────────────────────────

    def _column_aliases(self, col: Dict[str, Any]) -> List[str]:
        aliases: Set[str] = set()
        stop_tokens = {"gel", "peg", "phase", "column", "columns", "based",
                       "polymer", "polymeric", "silica"}
        for field in ("column_name", "modification"):
            val = col.get(field)
            if not val or not isinstance(val, str):
                continue
            v = normalize_for_parsing(val).lower()
            if not v:
                continue
            aliases.add(v)
            for tok in re.split(r"[^a-z0-9]+", v):
                if not tok or len(tok) < 3:
                    continue
                if tok.isdigit() or not any(ch.isalpha() for ch in tok):
                    continue
                if tok in stop_tokens:
                    continue
                aliases.add(tok)
        return sorted(aliases, key=len, reverse=True)

    def _column_primary_aliases(self, col: Dict[str, Any]) -> Set[str]:
        name = normalize_for_parsing(str(col.get("column_name") or "")).lower()
        tokens: Set[str] = set()
        for tok in re.split(r"[^a-z0-9]+", name):
            if len(tok) >= 4 and any(ch.isalpha() for ch in tok):
                tokens.add(tok)
        return tokens

    # ── evidence collection ───────────────────────────────────────────────────

    def _collect_column_evidence(
        self,
        paragraphs: List[str],
        catalog: List[Dict[str, Any]],
    ) -> Tuple[List[List[str]], List[str]]:
        if not paragraphs:
            return [], []

        norm_paras = [normalize_for_parsing(p).lower() for p in paragraphs]
        tech_idxs = {i for i, p in enumerate(norm_paras) if any(k in p for k in self.TECH_KEYWORDS)}
        tech_paras = [paragraphs[i] for i in sorted(tech_idxs)]

        primary_aliases = [self._column_primary_aliases(col) for col in catalog]
        spec_patterns = [
            re.compile(r"particle\s+(?:diameter|size)", re.I),
            re.compile(r"pore\s+size", re.I),
            re.compile(r"nominal\s+pore", re.I),
            re.compile(r"\b\d+\s*x\s*\d+(?:\.\d+)?\s*mm\b", re.I),
        ]

        per_column: List[List[str]] = []
        total = len(paragraphs)

        for col_idx, col in enumerate(catalog):
            aliases = self._column_aliases(col)
            other_aliases: Set[str] = set()
            for i, tokens in enumerate(primary_aliases):
                if i != col_idx:
                    other_aliases.update(tokens)

            idxs: Set[int] = set()
            if aliases:
                for i, p in enumerate(norm_paras):
                    if not any(alias in p for alias in aliases):
                        continue
                    if other_aliases and any(other in p for other in other_aliases):
                        continue
                    is_cap = any(solvent in p for solvent in self.SOLVENT_KEYWORDS)
                    is_spec = any(pat.search(p) for pat in spec_patterns)
                    if is_cap:
                        for j in range(i - self.EVIDENCE_WINDOW, i + self.EVIDENCE_WINDOW + 1):
                            if 0 <= j < total:
                                if not (other_aliases and any(other in norm_paras[j] for other in other_aliases)):
                                    idxs.add(j)
                    elif is_spec:
                        idxs.add(i)

            if not idxs and primary_aliases[col_idx]:
                for i, p in enumerate(norm_paras):
                    if not any(alias in p for alias in primary_aliases[col_idx]):
                        continue
                    if not (
                        any(s in p for s in self.SOLVENT_KEYWORDS)
                        or any(k in p for k in self.CAP_KEYWORDS)
                    ):
                        continue
                    if other_aliases and any(other in p for other in other_aliases):
                        continue
                    idxs.add(i)

            per_column.append([paragraphs[i] for i in sorted(idxs)])

        return per_column, tech_paras

    def _build_evidence_text(
        self,
        evidence: List[str],
        tech_paras: List[str],
        max_chars: int,
    ) -> str:
        seen: Set[str] = set()
        scored: List[Tuple[int, int, str]] = []
        for idx, p in enumerate(evidence):
            norm = normalize_for_parsing(p)
            if not norm or norm in seen:
                continue
            norm_l = norm.lower()
            score = 0
            if any(k in norm_l for k in self.CAP_KEYWORDS):
                score += 3
            if any(s in norm_l for s in self.SOLVENT_KEYWORDS):
                score += 1
            scored.append((score, idx, p))
            seen.add(norm)
        scored.sort(key=lambda x: (-x[0], x[1]))

        combined: List[str] = [p for _, _, p in scored]
        for p in tech_paras:
            norm = normalize_for_parsing(p)
            if norm and norm not in seen:
                seen.add(norm)
                combined.append(p)

        text = "\n\n".join(combined)
        if len(text) > max_chars:
            text = text[:max_chars]
        return self._sanitize_for_llm(text)

    def _build_global_evidence(self, paragraphs: List[str]) -> str:
        scored: List[Tuple[int, int, str]] = []
        for i, p in enumerate(paragraphs):
            score = self._score_paragraph(p)
            if score > 0:
                scored.append((score, i, p))
        scored.sort(key=lambda x: (-x[0], x[1]))

        selected: List[str] = []
        total_chars = 0
        for _, _, p in scored:
            if total_chars >= self.max_global_chars:
                break
            selected.append(p)
            total_chars += len(p) + 2

        if not selected:
            for p in paragraphs[:6]:
                if total_chars >= self.max_global_chars:
                    break
                selected.append(p)
                total_chars += len(p) + 2

        return self._sanitize_for_llm("\n\n".join(selected))

    def _condition_tokens(self, cond: Dict[str, Any]) -> List[str]:
        tokens: List[str] = []
        sp = cond.get("SP_FIELDS") or {}
        sol = cond.get("SOL_FIELDS") or {}
        for field in ("column_name", "material", "modification", "phase", "manufacturer"):
            val = sp.get(field)
            if not val:
                continue
            for tok in re.split(r"[^a-zA-Z0-9]+", normalize_for_parsing(str(val)).lower()):
                if len(tok) >= 3 and any(ch.isalpha() for ch in tok):
                    tokens.append(tok)
        solvent = sol.get("solvent") or ""
        for part in re.split(r"\s*/\s*|\s+and\s+", normalize_for_parsing(str(solvent)).lower()):
            if len(part) >= 3 and any(ch.isalpha() for ch in part):
                tokens.append(part)
        ratio = sol.get("ratio")
        if ratio:
            tokens.extend(re.findall(r"\d+(?:\.\d+)?", str(ratio)))
        return list(dict.fromkeys(tokens))

    def _build_condition_evidence(
        self, cond: Dict[str, Any], paragraphs: List[str], max_chars: int
    ) -> str:
        tokens = self._condition_tokens(cond)
        if not tokens:
            return ""
        scored: List[Tuple[int, int, str]] = []
        for idx, p in enumerate(paragraphs):
            t = normalize_for_parsing(p).lower()
            score = 0
            if any(tok in t for tok in tokens):
                score += 2
            if any(k in t for k in self.CAP_KEYWORDS):
                score += 2
            if any(s in t for s in self.SOLVENT_KEYWORDS):
                score += 1
            if re.search(r"\b\d+(?:\.\d+)?\s*(wt%|w/w|v/v|vol%|%)\b", t):
                score += 1
            if score > 0:
                scored.append((score, idx, p))
        scored.sort(key=lambda x: (-x[0], x[1]))
        selected: List[str] = []
        total_chars = 0
        for _, _, p in scored:
            if total_chars >= max_chars:
                break
            selected.append(p)
            total_chars += len(p) + 2
        return self._sanitize_for_llm("\n\n".join(selected))

    # ── prompt builders ───────────────────────────────────────────────────────

    def _build_prompt(self, chunk: LLMChunk) -> str:
        return f"""\
You are an expert scientific data extractor for polymer chromatography.

═══════════════════════════════════════════════════════════════════
STEP 1 — DECIDE WHETHER LCCC IS PRESENT
═══════════════════════════════════════════════════════════════════
LCCC (Liquid Chromatography at Critical Conditions) is present ONLY
when the text contains at least one of these signal phrases:

  • "critical conditions"
  • "critical adsorption point" / "CAP"
  • "LCCC" / "critical point of adsorption"
  • "chromatographically invisible"
  • "retention ... independent of molecular weight"
  • "adsorption-desorption transition"
  • "critical elution conditions"
  • "elutes like a [polymer] homopolymer" (said of a block copolymer)

If NONE of these phrases appear → return {{"LCCC_conditions":[]}} immediately.

═══════════════════════════════════════════════════════════════════
STEP 2 — UNDERSTAND WHAT MAKES A VALID LCCC CONDITION
═══════════════════════════════════════════════════════════════════
At the critical adsorption point:
  • ONE polymer block elutes at the SAME retention volume regardless
    of molar mass → it is "chromatographically invisible".
  • OTHER blocks separate normally by size.

Each DISTINCT experiment requires its OWN JSON object. Distinctness is
defined by: (polymer_system + column + mobile phase ratio + temperature).

⚠ SERIES OF CONDITIONS: Papers often optimize conditions by varying
  the mobile phase ratio or temperature for different polymer architectures
  (e.g. diblock vs triblock, short blocks vs long blocks). Treat each
  distinct combination as a SEPARATE condition. Never collapse them.

  Example: "60:40 ACN/water at 35 C for PEO-b-PLLA diblock" and
           "54:46 ACN/water at 54 C for PLLA-b-PEO-b-PLLA triblock"
  → TWO separate conditions even though they use the same column.

═══════════════════════════════════════════════════════════════════
STEP 3 — EXTRACT EACH CONDITION
═══════════════════════════════════════════════════════════════════
For each distinct LCCC experiment found, populate this schema:

{_SCHEMA}

{_FIELD_GUIDE}

═══════════════════════════════════════════════════════════════════
STEP 4 — SELF-AUDIT (mandatory before writing your answer)
═══════════════════════════════════════════════════════════════════
{_ANTI_HALLUCINATION}

═══════════════════════════════════════════════════════════════════
{_OUTPUT_FORMAT}
═══════════════════════════════════════════════════════════════════
TEXT
═══════════════════════════════════════════════════════════════════
{chunk.llm_text}
"""

    def _build_prompt_with_catalog(self, chunk: LLMChunk, catalog: List[Dict[str, Any]]) -> str:
        base = self._build_prompt(chunk)
        if not catalog:
            return base
        lines: List[str] = []
        for col in catalog[:8]:
            parts = [f"{f}={v}" for f in SP_FIELDS if (v := col.get(f))]
            if parts:
                lines.append("  - " + "; ".join(parts))
        if lines:
            catalog_block = (
                "\n═══════════════════════════════════════════════════════════════════\n"
                "KNOWN COLUMNS IN THIS PAPER\n"
                "Use these ONLY to fill SP_FIELDS when the column is CONFIRMED by\n"
                "the evidence text. Do NOT assign catalog values to a condition\n"
                "unless the column is explicitly named in the evidence.\n"
                "═══════════════════════════════════════════════════════════════════\n"
                + "\n".join(lines)
                + "\n"
            )
            base = base + catalog_block
        return base

    def _build_column_prompt(self, column: Dict[str, Any], evidence_text: str) -> str:
        col_name = column.get("column_name") or "(unknown)"
        col_desc = "; ".join(f"{f}={v}" for f in SP_FIELDS if (v := column.get(f)))
        return f"""\
You are an expert scientific data extractor for polymer chromatography.

═══════════════════════════════════════════════════════════════════
TARGET COLUMN (extract ONLY conditions that use THIS column)
═══════════════════════════════════════════════════════════════════
Name : {col_name}
Spec : {col_desc or "see evidence text"}

⚠ MATCHING RULE: Only extract a condition if this column is EXPLICITLY
  named in the evidence text for that condition. If this column is not
  mentioned in the evidence → return {{"LCCC_conditions":[]}} immediately.
  Do NOT infer the column from context.

═══════════════════════════════════════════════════════════════════
TASK
═══════════════════════════════════════════════════════════════════
Extract ALL LCCC conditions that use the target column above.

One column may appear in multiple experiments (different polymer
architectures, different mobile phase ratios, different temperatures).
Each is a SEPARATE condition. If the paper tests N polymer systems on
this column → return N condition objects.

═══════════════════════════════════════════════════════════════════
SCHEMA
═══════════════════════════════════════════════════════════════════
{_SCHEMA}

═══════════════════════════════════════════════════════════════════
WHAT TO FIND FOR EACH CONDITION
═══════════════════════════════════════════════════════════════════
A. critical_polymer_unit
   Which polymer block is chromatographically invisible on this column?
   ⚠ This field is REQUIRED if the text states it.
   It is always the block whose retention is "independent of molecular weight".
   ⚠ If not stated: null. Do NOT infer.

B. SOL_FIELDS — mobile phase
   solvent    : names slash-separated as written
   ratio      : BOTH parts exactly as written ("60/40", "54:46")
   ratio_units: "v/v"  "wt%"  "vol%"

C. TECH_FIELDS — temperature
   ⚠⚠ Single numeric value ONLY (strip the unit symbol).
      "35 C" → "35"   "54 degC" → "54"
   ⚠⚠ If the text only gives a RANGE (e.g. "6-54 C") → null.
   ⚠⚠ ISOLATION: Do NOT borrow a temperature from a different condition.
      Assign only the temperature paired with THIS specific mobile phase.

D. Other TECH_FIELDS: flow_rate, injection_volume,
   injected_polymer_concentration, injected_polymer_solvent_solution.

E. separation_behavior
   critical_block_behavior    : full sentence: "[block] elutes/becomes..."
   non_critical_block_behavior: full sentence describing the other block
   purpose                    : why LCCC is applied

{_FIELD_GUIDE}

{_ANTI_HALLUCINATION}

{_OUTPUT_FORMAT}

═══════════════════════════════════════════════════════════════════
EVIDENCE TEXT
═══════════════════════════════════════════════════════════════════
{evidence_text}
"""

    def _build_global_prompt(self, evidence_text: str) -> str:
        return f"""\
You are an expert scientific data extractor for polymer chromatography.

═══════════════════════════════════════════════════════════════════
TASK — Comprehensive LCCC extraction
═══════════════════════════════════════════════════════════════════
Extract EVERY experimentally described LCCC condition in the text.
This is a high-recall pass: capture all distinct combinations.

═══════════════════════════════════════════════════════════════════
WHAT IS LCCC?
═══════════════════════════════════════════════════════════════════
At the critical adsorption point, one polymer block elutes at identical
retention volume regardless of molar mass — it is "chromatographically
invisible". All other blocks separate normally.

Signal phrases that confirm LCCC is present:
  • "critical conditions" / "critical adsorption point" / "CAP"
  • "LCCC" / "critical point"
  • "chromatographically invisible"
  • "retention independent of molecular weight"
  • "adsorption-desorption transition"
  • "elutes like a [polymer] homopolymer" (for a block copolymer)

═══════════════════════════════════════════════════════════════════
HOW TO COUNT DISTINCT CONDITIONS
═══════════════════════════════════════════════════════════════════
Each unique combination of:
  (polymer_system + column + mobile phase ratio + temperature)
is ONE condition. Every combination gets its OWN JSON object.

⚠ SERIES OF CONDITIONS — the most common mistake is collapsing a series
  into a single record. Watch for these patterns:

  Pattern A — Same column, different polymer architectures:
    "For PEO-b-PLLA diblock, critical conditions were 60:40 ACN/water
     at 35 C. For PLLA-b-PEO-b-PLLA triblock, critical conditions were
     54:46 ACN/water at 54 C."
    → TWO conditions. Different polymer_system, ratio, and temperature.

  Pattern B — Same column, varying ratio to find the critical point:
    "Solvent ratios of 56/44, 57/43, and 58/42 were tested at 37 C."
    → Only extract ratios explicitly paired with a LCCC signal phrase.

  Pattern C — Same ratio, different temperatures:
    "At 35 C the PEO block was critical; at 54 C the PMMA block was critical."
    → TWO conditions with different critical_polymer_unit and temperature.

═══════════════════════════════════════════════════════════════════
SCHEMA
═══════════════════════════════════════════════════════════════════
{_SCHEMA}

{_FIELD_GUIDE}

{_ANTI_HALLUCINATION}

{_OUTPUT_FORMAT}

═══════════════════════════════════════════════════════════════════
TEXT
═══════════════════════════════════════════════════════════════════
{evidence_text}
"""

    def _build_refine_prompt(self, condition: Dict[str, Any], evidence_text: str) -> str:
        # Serialize current non-null values clearly so the LLM knows what to protect
        existing_values: List[str] = []
        for top_field in ("critical_polymer_unit", "polymer_system"):
            v = condition.get(top_field)
            if v is not None:
                existing_values.append(f"  {top_field}: {v!r}  ← DO NOT CHANGE")
        for section in ("SP_FIELDS", "SOL_FIELDS", "TECH_FIELDS", "separation_behavior"):
            sec = condition.get(section) or {}
            for k, v in sec.items():
                if v is not None:
                    existing_values.append(f"  {section}.{k}: {v!r}  ← DO NOT CHANGE")
        locked_block = "\n".join(existing_values) if existing_values else "  (none yet)"

        return f"""\
You are refining an already-extracted LCCC condition record.
Your ONLY job is to fill null fields. Existing values are LOCKED.

═══════════════════════════════════════════════════════════════════
LOCKED VALUES — you must reproduce these EXACTLY, unchanged
═══════════════════════════════════════════════════════════════════
{locked_block}

═══════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════
1. Search the evidence text for values for the NULL fields only.
2. If found explicitly → fill the field following the rules below.
3. If NOT found → leave null.
4. Return ONE complete condition object in the schema below.

Fields most likely to have values in the evidence text at this stage:
  • pore_size, particle_diameter (look for A, um, Angstrom, micron)
  • column_dimensions (look for "mm" with two numbers)
  • manufacturer (company name near the column name)
  • flow_rate (look for mL/min, ml/min)
  • injection_volume (look for uL, ul, microL)
  • injected_polymer_concentration (look for mg/mL, g/L)
  • separation_behavior fields (sentences about critical/non-critical blocks)

Temperature rule:
  If temperature is null and you find a SINGLE numeric value → fill it.
  If you find only a RANGE (e.g. "6-54 C", "20 to 40 C°") → leave null.

Ratio rule: always capture BOTH parts (e.g. "60/40, "54:46").

═══════════════════════════════════════════════════════════════════
SCHEMA
═══════════════════════════════════════════════════════════════════
{_SCHEMA}

{_FIELD_GUIDE}

{_ANTI_HALLUCINATION}

{_OUTPUT_FORMAT}

═══════════════════════════════════════════════════════════════════
EVIDENCE TEXT
═══════════════════════════════════════════════════════════════════
{evidence_text}
"""

    # ── field validation ──────────────────────────────────────────────────────

    def _filter_fields(
        self,
        data: Dict[str, Any],
        fields: List[str],
        text: str,
        soft_fields: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate and clean each field value against the source text.

        Philosophy:
        - Trust the LLM for string/label fields (names, phrases, descriptions)
          that are too diverse for strict text matching.
        - For numeric fields, verify the numbers appear in the text.
        - Temperature is always run through _sanitize_temperature.
        - Behavior sentence fields require a minimum length to reject one-word
          hallucinations.
        """
        soft_fields = soft_fields or set()
        t_norm = normalize_for_parsing(text).lower()

        # Fields that are trusted as long as they are non-empty strings and
        # are not suspiciously long (which would indicate hallucination).
        ALWAYS_TRUSTED_STRING_FIELDS = {
            "solvent", "phase", "modification", "material", "manufacturer",
            "column_name", "injected_polymer_solvent_solution",
        }

        # Behavior fields need to be full sentences; single words/phrases are
        # rejected because the prompt demands subject + verb + behavior.
        BEHAVIOR_FIELDS = {
            "critical_block_behavior",
            "non_critical_block_behavior",
            "purpose",
        }
        # Minimum word count for a behavior field to be accepted
        BEHAVIOR_MIN_WORDS = 5

        for f in fields:
            v = data.get(f)
            if v is None:
                continue

            # Coerce to usable type
            if isinstance(v, (int, float)):
                raw: Any = int(v) if isinstance(v, float) and float(v).is_integer() else v
                v_str = str(raw)
            elif isinstance(v, str):
                raw = v.strip()
                v_str = raw
            else:
                data[f] = None
                continue

            if not v_str and raw != 0:
                data[f] = None
                continue

            # Temperature: enforce single-value rule
            if f == "temperature":
                data[f] = self._sanitize_temperature(raw)
                continue

            # number_of_columns: always trust integer values
            if f == "number_of_columns":
                data[f] = raw
                continue

            # Behavior fields: require minimum sentence length
            if f in BEHAVIOR_FIELDS:
                word_count = len(str(raw).split())
                if word_count >= BEHAVIOR_MIN_WORDS and len(str(raw)) <= 500:
                    data[f] = raw
                else:
                    data[f] = None
                continue

            # Always-trusted string fields (names, labels)
            if f in ALWAYS_TRUSTED_STRING_FIELDS or f in soft_fields:
                if len(str(raw)) <= 300:
                    data[f] = raw
                else:
                    data[f] = None
                continue

            # Exact text match works for most remaining fields
            if value_in_text(v_str, text):
                data[f] = raw
                continue

            # Numeric fields: accept if all component numbers appear in text
            if f in {"ratio", "flow_rate", "injection_volume"}:
                if self._numbers_present(v_str, text):
                    data[f] = raw
                    continue

            # injected_polymer_concentration: numbers must appear
            if f == "injected_polymer_concentration":
                if self._numbers_present(v_str, text):
                    data[f] = raw
                    continue

            # ratio_units: accept if any known unit token appears
            if f == "ratio_units":
                if any(tok in t_norm for tok in ("wt%", "w/w", "v/v", "vol%", "%")):
                    data[f] = raw
                    continue

            # pore_size / particle_diameter: numbers must appear
            if f in {"pore_size", "particle_diameter"}:
                if self._numbers_present(v_str, text):
                    data[f] = raw
                    continue

            # column_dimensions: numbers must appear
            if f == "column_dimensions":
                if self._numbers_present(v_str, text):
                    data[f] = raw
                    continue

            data[f] = None
        return data

    # ── catalog helpers ───────────────────────────────────────────────────────

    def _apply_catalog(
        self, sp_clean: Dict[str, Any], catalog: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not catalog:
            return sp_clean

        def _match(sp: Dict[str, Any], col: Dict[str, Any]) -> bool:
            for f in ("column_name", "phase", "modification", "material"):
                a, b = sp.get(f), col.get(f)
                if not a or not b:
                    continue
                a_n = normalize_for_parsing(str(a)).lower()
                b_n = normalize_for_parsing(str(b)).lower()
                if a_n and b_n and (a_n in b_n or b_n in a_n):
                    return True
            return False

        for col in catalog:
            if _match(sp_clean, col):
                for f in SP_FIELDS:
                    if sp_clean.get(f) is None and col.get(f) is not None:
                        sp_clean[f] = col.get(f)
                break
        return sp_clean

    def _matches_catalog(
        self, sp_clean: Dict[str, Any], catalog: List[Dict[str, Any]]
    ) -> bool:
        if not catalog:
            return True

        def _match(sp: Dict[str, Any], col: Dict[str, Any]) -> bool:
            for f in ("column_name", "phase", "modification", "material"):
                a, b = sp.get(f), col.get(f)
                if not a or not b:
                    continue
                a_n = normalize_for_parsing(str(a)).lower()
                b_n = normalize_for_parsing(str(b)).lower()
                if a_n and b_n and (a_n in b_n or b_n in a_n):
                    return True
            return False

        return any(_match(sp_clean, col) for col in catalog)

    def _match_catalog_by_name(
        self, name: Optional[str], catalog: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not name or not catalog:
            return None
        name_norm = normalize_for_parsing(str(name)).lower()
        for col in catalog:
            col_name = normalize_for_parsing(str(col.get("column_name") or "")).lower()
            if col_name and (name_norm in col_name or col_name in name_norm):
                return col
        return None

    # ── condition normalisation ───────────────────────────────────────────────

    def _normalize_condition(
        self,
        cond: Dict[str, Any],
        validation_text: str,
        catalog: List[Dict[str, Any]],
        context_snippet: str,
        column_hint: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(cond, dict):
            return None

        # polymer_system
        polymer_system_raw = cond.get("polymer_system") or cond.get("Polymer_System")
        polymer_system: Optional[str] = None
        if isinstance(polymer_system_raw, str) and polymer_system_raw.strip():
            polymer_system = polymer_system_raw.strip()

        # critical_polymer_unit
        unit_raw = cond.get("critical_polymer_unit") or cond.get("critical_unit")
        if isinstance(unit_raw, dict):
            unit_raw = next(iter(unit_raw.keys()), None) or next(iter(unit_raw.values()), None)
        unit: Optional[str] = None
        if isinstance(unit_raw, str) and unit_raw.strip():
            unit = unit_raw.strip()

        # NOTE: The fallback unit inference that was previously here has been
        # removed. It was a hallucination risk: guessing the critical polymer
        # unit from context without explicit text evidence produced incorrect
        # values that then contaminated downstream merging. If the LLM did not
        # return a unit, we leave it null and rely on the refinement pass or
        # the merge step to fill it from a companion extraction that got it right.

        # SP_FIELDS
        sp_raw = cond.get("SP_FIELDS") or cond.get("stationary_phase") or {}
        if isinstance(sp_raw, list):
            sp_raw = sp_raw[0] if sp_raw else {}
        if not isinstance(sp_raw, dict):
            sp_raw = {}

        sp_clean = {f: sp_raw.get(f) for f in SP_FIELDS}
        sp_clean = self._filter_fields(
            sp_clean, SP_FIELDS, validation_text,
            soft_fields={"column_name", "phase", "modification", "material", "manufacturer"},
        )
        if column_hint:
            for f in SP_FIELDS:
                if sp_clean.get(f) is None and column_hint.get(f) is not None:
                    sp_clean[f] = column_hint.get(f)
        sp_clean = self._apply_catalog(sp_clean, catalog)
        if not self._matches_catalog(sp_clean, catalog):
            return None

        # SOL_FIELDS
        sol_raw = cond.get("SOL_FIELDS") or cond.get("solvent_details") or {}
        if isinstance(sol_raw, list):
            sol_raw = sol_raw[0] if sol_raw else {}
        if not isinstance(sol_raw, dict):
            sol_raw = {}

        sol_clean = {f: sol_raw.get(f) for f in SOL_FIELDS}
        sol_clean = self._filter_fields(
            sol_clean, SOL_FIELDS, validation_text,
            soft_fields={"solvent"},
        )

        # NOTE: The solvent keyword fallback that was previously here has been
        # removed. If two solvents appear in context but neither is explicitly
        # tied to THIS condition, inferring them risks cross-contamination
        # (e.g. assigning a solvent from a different experimental block).
        # Leave solvent null; the global pass and refinement pass will fill it
        # from text that explicitly associates the solvent with this condition.

        # TECH_FIELDS
        tech_raw = cond.get("TECH_FIELDS") or cond.get("technical_details") or {}
        if isinstance(tech_raw, list):
            tech_raw = tech_raw[0] if tech_raw else {}
        if not isinstance(tech_raw, dict):
            tech_raw = {}

        tech_clean = {f: tech_raw.get(f) for f in TECH_FIELDS}
        tech_clean = self._filter_fields(
            tech_clean, TECH_FIELDS, validation_text,
            soft_fields={"injected_polymer_solvent_solution", "injected_polymer_concentration"},
        )

        # separation_behavior — behavior sentence fields are validated in
        # _filter_fields with a minimum word count check
        sep_raw = cond.get("separation_behavior") or cond.get("behavior") or {}
        if not isinstance(sep_raw, dict):
            sep_raw = {}
        sep_clean = {f: sep_raw.get(f) for f in SEP_FIELDS}
        sep_clean = self._filter_fields(sep_clean, SEP_FIELDS, validation_text)

        has_data = (
            unit is not None
            or any(v is not None for v in sp_clean.values())
            or any(v is not None for v in sol_clean.values())
            or any(v is not None for v in tech_clean.values())
            or any(v is not None for v in sep_clean.values())
        )
        if not has_data:
            return None

        return {
            "critical_polymer_unit": unit,
            "polymer_system": polymer_system,
            "SP_FIELDS": sp_clean,
            "SOL_FIELDS": sol_clean,
            "TECH_FIELDS": tech_clean,
            "separation_behavior": sep_clean,
            "context_snippet": context_snippet,
        }

    # ── deduplication and merging ─────────────────────────────────────────────

    def _cond_key(self, cond: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
        """
        Uniquely identify a condition by six dimensions:
          (unit, polymer_system, column, solvent, ratio, temperature)

        Rules:
        - Empty string represents null (field not yet known).
        - A non-empty value in one record and empty in another does NOT prevent
          a merge (the null is a missing observation, not a contradiction).
        - Two non-empty values that DIFFER always mean distinct conditions.
        """
        unit = normalize_for_parsing(str(cond.get("critical_polymer_unit") or "")).lower().strip()
        psys = normalize_for_parsing(str(cond.get("polymer_system") or "")).lower().strip()
        sp = cond.get("SP_FIELDS") or {}
        sol = cond.get("SOL_FIELDS") or {}
        tech = cond.get("TECH_FIELDS") or {}
        col = normalize_for_parsing(str(sp.get("column_name") or "")).lower().strip()
        solvent = normalize_for_parsing(str(sol.get("solvent") or "")).lower().strip()
        ratio = normalize_for_parsing(str(sol.get("ratio") or "")).lower().strip()
        temp = normalize_for_parsing(str(tech.get("temperature") or "")).lower().strip()
        return (unit, psys, col, solvent, ratio, temp)

    def _conditions_are_same_experiment(
        self, a: Dict[str, Any], b: Dict[str, Any]
    ) -> bool:
        """
        Return True only if a and b describe the same experiment.

        Merge logic:
        - For each key dimension, if BOTH records have a non-empty value
          and those values DIFFER → different experiments, do not merge.
        - If one or both records have an empty (null) value for a dimension
          → that dimension is not a differentiator (missing observation).

        This means: two conditions with the same ratio but one missing a
        temperature and the other having a temperature CAN merge. But two
        conditions with ratio "60/40" and "54:46" will NEVER merge.
        """
        ka = self._cond_key(a)
        kb = self._cond_key(b)

        for da, db in zip(ka, kb):
            if da and db and da != db:
                return False  # Both populated and they differ → distinct conditions

        return True

    def _merge_conditions(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge b into a: fill null top-level and section fields in a from b.
        Never overwrites a non-null field.
        """
        merged = dict(a)
        for top_field in ("critical_polymer_unit", "polymer_system"):
            if not merged.get(top_field) and b.get(top_field):
                merged[top_field] = b[top_field]
        for section in ("SP_FIELDS", "SOL_FIELDS", "TECH_FIELDS", "separation_behavior"):
            a_sec = dict(a.get(section) or {})
            b_sec = b.get(section) or {}
            for k, v in b_sec.items():
                if a_sec.get(k) is None and v is not None:
                    a_sec[k] = v
            merged[section] = a_sec
        return merged

    def _deduplicate_and_merge(
        self, conditions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Group conditions that describe the same experiment and merge each group
        into one maximally-populated record.
        """
        result: List[Dict[str, Any]] = []
        for cond in conditions:
            merged_into = None
            for existing in result:
                if self._conditions_are_same_experiment(existing, cond):
                    merged_into = existing
                    break
            if merged_into is not None:
                idx = result.index(merged_into)
                result[idx] = self._merge_conditions(merged_into, cond)
            else:
                result.append(dict(cond))
        return result

    def _apply_consensus(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.consensus_min <= 1:
            for c in conditions:
                c.pop("_model", None)
            return self._deduplicate_and_merge(conditions)

        support: Dict[Tuple[str, str, str, str, str, str], set] = {}
        for cond in conditions:
            key = self._cond_key(cond)
            model = cond.get("_model") or ""
            if model:
                support.setdefault(key, set()).add(model)

        filtered: List[Dict[str, Any]] = []
        for cond in conditions:
            if len(support.get(self._cond_key(cond), set())) >= self.consensus_min:
                cond.pop("_model", None)
                filtered.append(cond)
        return self._deduplicate_and_merge(filtered)

    # ── LLM output parser ─────────────────────────────────────────────────────

    def _parse_conditions(
        self, outputs: List[Tuple[str, Any]], stats: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        results: List[Tuple[str, Dict[str, Any]]] = []
        for model_name, data in outputs:
            if not data:
                stats["parse_failures"] += 1
                continue
            conditions: Any = None
            if isinstance(data, dict):
                conditions = data.get("LCCC_conditions") or data.get("conditions")
            elif isinstance(data, list):
                conditions = data
            if not isinstance(conditions, list):
                stats["parse_failures"] += 1
                continue
            for cond in conditions:
                if isinstance(cond, dict):
                    results.append((model_name, cond))
        return results

    # ── main extraction entry point ───────────────────────────────────────────

    def extract(
        self,
        paragraphs: List[str],
        full_text: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not self.available:
            return [], {"llm_available": False}

        catalog = self._extract_columns_catalog(paragraphs, full_text)
        max_chars = max(2000, int(self.config.max_context_chars))

        extracted: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "llm_available": True,
            "chunks": 0,
            "llm_calls": 0,
            "conditions": 0,
            "parse_failures": 0,
            "mode": "dynamic",
            "columns_catalog": len(catalog),
        }

        # ── Pass 1: per-column targeted extraction ────────────────────────────
        if catalog:
            per_column_evidence, tech_paras = self._collect_column_evidence(paragraphs, catalog)
            if any(per_column_evidence):
                stats["chunks"] += len(per_column_evidence)
                stats["mode"] = "per_column+dynamic" if self.high_accuracy else "per_column"
                for idx, col in enumerate(catalog):
                    evidence = per_column_evidence[idx] if idx < len(per_column_evidence) else []
                    if not evidence:
                        continue
                    evidence_text = self._build_evidence_text(evidence, tech_paras, max_chars)
                    if not evidence_text:
                        continue

                    prompt = self._build_column_prompt(col, evidence_text)
                    outputs = self.client.run_json(prompt)
                    stats["llm_calls"] += max(1, len(self.models))
                    if self.debug:
                        logger.info(f"LLM column {idx+1}/{len(catalog)} (chars={len(evidence_text)})")

                    for model_name, cond in self._parse_conditions(outputs or [], stats):
                        normalized = self._normalize_condition(
                            cond,
                            validation_text=evidence_text,
                            catalog=catalog,
                            context_snippet=normalize_for_parsing(evidence_text)[:1200],
                            column_hint=col,
                        )
                        if normalized:
                            normalized["_model"] = model_name
                            extracted.append(normalized)

                extracted = self._apply_consensus(extracted)
                logger.info(f"LLM per-column pass: {len(extracted)} conditions")

        # ── Pass 2: global high-recall pass ───────────────────────────────────
        if self.high_accuracy:
            global_text = self._build_global_evidence(paragraphs)
            if global_text:
                prompt = self._build_global_prompt(global_text)
                outputs = self.client.run_json(prompt)
                stats["llm_calls"] += max(1, len(self.models))

                for model_name, cond in self._parse_conditions(outputs or [], stats):
                    normalized = self._normalize_condition(
                        cond,
                        validation_text=global_text,
                        catalog=catalog,
                        context_snippet=normalize_for_parsing(global_text)[:1200],
                    )
                    if normalized:
                        normalized["_model"] = model_name
                        extracted.append(normalized)

        # ── Pass 3: sliding-window chunk pass ─────────────────────────────────
        chunks = self.build_chunks(paragraphs)
        stats["chunks"] += len(chunks)
        for chunk in chunks:
            prompt = self._build_prompt_with_catalog(chunk, catalog)
            outputs = self.client.run_json(prompt)
            stats["llm_calls"] += max(1, len(self.models))
            if self.debug:
                logger.info(
                    f"LLM chunk {chunk.chunk_id}/{len(chunks)} "
                    f"paras {chunk.para_start}-{chunk.para_end} "
                    f"(raw={len(chunk.raw_text)}, llm={len(chunk.llm_text)})"
                )
            for model_name, cond in self._parse_conditions(outputs or [], stats):
                normalized = self._normalize_condition(
                    cond,
                    validation_text=full_text or chunk.raw_text,
                    catalog=catalog,
                    context_snippet=normalize_for_parsing(chunk.raw_text)[:1200],
                )
                if normalized:
                    normalized["_model"] = model_name
                    extracted.append(normalized)

        extracted = self._apply_consensus(extracted)

        # ── Pass 4: refinement pass ───────────────────────────────────────────
        if self.high_accuracy and self.refine_enabled and extracted:
            refined: List[Dict[str, Any]] = []
            for cond in extracted[: self.refine_limit]:
                sp = cond.get("SP_FIELDS") or {}
                sol = cond.get("SOL_FIELDS") or {}
                tech = cond.get("TECH_FIELDS") or {}
                sep = cond.get("separation_behavior") or {}
                needs_refine = (
                    any(v is None for v in sp.values())
                    or any(v is None for v in sol.values())
                    or any(v is None for v in tech.values())
                    or any(v is None for v in sep.values())
                )
                if not needs_refine:
                    refined.append(cond)
                    continue

                evidence_text = self._build_condition_evidence(cond, paragraphs, max_chars=max_chars)
                if not evidence_text:
                    refined.append(cond)
                    continue

                prompt = self._build_refine_prompt(cond, evidence_text)
                outputs = self.client.run_json(prompt)
                stats["llm_calls"] += max(1, len(self.models))

                updated = None
                for model_name, raw_cond in self._parse_conditions(outputs or [], stats):
                    column_hint = self._match_catalog_by_name(sp.get("column_name"), catalog)
                    normalized = self._normalize_condition(
                        raw_cond,
                        validation_text=evidence_text,
                        catalog=catalog,
                        context_snippet=normalize_for_parsing(evidence_text)[:1200],
                        column_hint=column_hint,
                    )
                    if normalized:
                        updated = normalized
                        break
                refined.append(updated or cond)

            if len(extracted) > self.refine_limit:
                refined.extend(extracted[self.refine_limit:])
            extracted = refined

        # Final dedup + merge across all passes
        extracted = self._deduplicate_and_merge(extracted)

        stats["conditions"] = len(extracted)
        stats["consensus_min"] = self.consensus_min
        stats["models"] = list(self.models)
        logger.info(f"LLM extraction complete: {len(extracted)} conditions")
        return extracted, stats