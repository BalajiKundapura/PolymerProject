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


LCCC_FIELDS = ["critical_polymer_unit", "SP_FIELDS", "SOL_FIELDS", "TECH_FIELDS", "separation_behavior"]
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


@dataclass
class LLMChunk:
    chunk_id: int
    para_start: int
    para_end: int
    raw_text: str
    llm_text: str


class LLMExtractor:
    WINDOW_PARAGRAPHS = 2
    OVERLAP_PARAGRAPHS = 1
    EVIDENCE_WINDOW = 0
    SOLVENT_KEYWORDS = [
        "acetone",
        "water",
        "methanol",
        "ethanol",
        "propanol",
        "isopropanol",
        "acetonitrile",
        "butanone",
        "methyl ethyl ketone",
        "cyclohexane",
        "hexane",
        "heptane",
        "toluene",
        "chloroform",
        "dichloromethane",
        "thf",
        "tetrahydrofuran",
        "dmf",
        "dmso",
        "buffer",
        "phosphate",
    ]
    KEYWORDS = [
        "column",
        "columns",
        "cap",
        "critical",
        "critical conditions",
        "critical adsorption",
        "lccc",
        "mobile phase",
        "eluent",
        "acetone",
        "water",
        "methanol",
        "acetonitrile",
        "butanone",
        "cyclohexane",
        "thf",
        "tetrahydrofuran",
        "wt%",
        "w/w",
        "v/v",
        "vol%",
        "flow rate",
        "temperature",
        "injection",
        "diol",
        "jordi",
        "nucleosil",
        "hs-peg",
        "discovery",
    ]
    CAP_KEYWORDS = [
        "cap",
        "critical",
        "critical conditions",
        "critical adsorption",
        "critical point",
        "lccc",
    ]
    TECH_KEYWORDS = [
        "temperature",
        "flow rate",
        "flow-rate",
        "injection",
        "loop",
        "ml/min",
        "mL/min",
        "autosampler",
        "system a",
        "system b",
    ]
    SECTION_HINTS = [
        "experimental",
        "materials",
        "methods",
        "results",
        "discussion",
        "conclusions",
    ]

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.client = LLMClient(config)
        self.config = self.client.config
        self.available = self.client.available
        self.models = self.client.models
        self.consensus_min = max(1, int(self.config.consensus_min or 1))
        self.high_accuracy = os.getenv("LCCC_LLM_HIGH_ACCURACY", "1").strip().lower() in ("1", "true", "yes")
        self.max_global_chars = int(os.getenv("LCCC_LLM_GLOBAL_CHARS", "6000"))
        self.refine_enabled = os.getenv("LCCC_LLM_REFINE", "1").strip().lower() in ("1", "true", "yes")
        self.refine_limit = max(1, int(os.getenv("LCCC_LLM_REFINE_LIMIT", "6")))
        self.debug = os.getenv("LCCC_LLM_DEBUG", "0").strip().lower() in ("1", "true", "yes")

    def _sanitize_for_llm(self, text: str) -> str:
        if not text:
            return ""
        t = normalize_for_parsing(text)
        # Remove citation brackets like [1], [1-3], [1,2,5]
        t = re.sub(r"\[(?:\d+|[,\-\s]){1,20}\]", " ", t)
        # Remove figure/table references that add noise
        t = re.sub(r"\b(?:fig(?:ure)?|table)\s*\d+[a-z]?\b", " ", t, flags=re.I)
        # Normalize whitespace
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def _numbers_present(self, value: str, text: str) -> bool:
        nums = re.findall(r"\d+(?:\.\d+)?", value or "")
        if not nums:
            return False
        t = normalize_for_parsing(text)
        return all(n in t for n in nums)

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
        total_paras = len(paragraphs)
        while idx <= end and idx < total_paras:
            chunk_start = idx
            chunk_end = idx
            length = 0
            truncated_text = None
            while chunk_end <= end and chunk_end < total_paras:
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

            if truncated_text is not None:
                text = truncated_text
            else:
                text = "\n\n".join(paragraphs[chunk_start:chunk_end])
            chunks.append((chunk_start, chunk_end - 1, text))

            if chunk_end > end:
                break

            next_start = max(chunk_end - overlap_paragraphs, chunk_start + 1)
            if next_start <= chunk_start:
                next_start = chunk_start + 1
            idx = next_start

        return chunks

    def _is_relevant(self, text: str) -> bool:
        t = (text or "").lower()
        return any(k in t for k in self.KEYWORDS)

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
        # Merge overlapping windows
        windows.sort(key=lambda w: (w[0], w[1]))
        merged: List[Tuple[int, int]] = []
        for start, end in windows:
            if not merged:
                merged.append((start, end))
                continue
            last_start, last_end = merged[-1]
            if start <= last_end + 1:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        chunks: List[LLMChunk] = []
        chunk_id = 0
        max_chars = max(400, int(self.config.max_context_chars))

        for start, end in merged:
            for c_start, c_end, text in self._chunk_window(
                paragraphs, start, end, max_chars=max_chars, overlap_paragraphs=self.OVERLAP_PARAGRAPHS
            ):
                llm_text = self._sanitize_for_llm(text)
                if not llm_text:
                    continue
                if not self._is_relevant(llm_text):
                    continue

                chunk_id += 1
                chunks.append(
                    LLMChunk(
                        chunk_id=chunk_id,
                        para_start=c_start,
                        para_end=c_end,
                        raw_text=text,
                        llm_text=llm_text,
                    )
                )

        return chunks

    def _column_aliases(self, col: Dict[str, Any]) -> List[str]:
        aliases: Set[str] = set()
        stop_tokens = {
            "gel",
            "peg",
            "phase",
            "column",
            "columns",
            "based",
            "polymer",
            "polymeric",
            "silica",
        }
        for field in ("column_name", "modification"):
            val = col.get(field)
            if not val or not isinstance(val, str):
                continue
            v = normalize_for_parsing(val).lower()
            if not v:
                continue
            aliases.add(v)
            if "hs-peg" in v or "hs peg" in v:
                aliases.add("hs-peg")
            for tok in re.split(r"[^a-z0-9]+", v):
                if not tok or len(tok) < 3:
                    continue
                if tok.isdigit():
                    continue
                if not any(ch.isalpha() for ch in tok):
                    continue
                if tok in stop_tokens:
                    continue
                aliases.add(tok)

        if "hs" in aliases and "peg" in aliases:
            aliases.add("hs-peg")
        if "dvb" in aliases:
            aliases.add("divinylbenzene")

        name = normalize_for_parsing(str(col.get("column_name") or "")).lower()
        if "nucleosil" in name:
            aliases.add("diol")

        return sorted(aliases, key=len, reverse=True)

    def _column_primary_aliases(self, col: Dict[str, Any]) -> Set[str]:
        name = normalize_for_parsing(str(col.get("column_name") or "")).lower()
        tokens: Set[str] = set()
        if "hs-peg" in name or "hs peg" in name:
            tokens.add("hs-peg")
        for tok in re.split(r"[^a-z0-9]+", name):
            if len(tok) < 4:
                continue
            if not any(ch.isalpha() for ch in tok):
                continue
            tokens.add(tok)
        if "nucleosil" in name:
            tokens.add("nucleosil")
            tokens.add("diol")
        return tokens

    def _collect_column_evidence(
        self, paragraphs: List[str], catalog: List[Dict[str, Any]]
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
                    if any(alias in p for alias in aliases):
                        if other_aliases and any(other in p for other in other_aliases):
                            continue
                        is_cap = any(solvent in p for solvent in self.SOLVENT_KEYWORDS)
                        is_spec = any(pattern.search(p) for pattern in spec_patterns)
                        if is_cap:
                            for j in range(i - self.EVIDENCE_WINDOW, i + self.EVIDENCE_WINDOW + 1):
                                if 0 <= j < total:
                                    if not (other_aliases and any(other in norm_paras[j] for other in other_aliases)):
                                        idxs.add(j)
                        elif is_spec:
                            idxs.add(i)

            if not idxs:
                for i, p in enumerate(norm_paras):
                    if not primary_aliases[col_idx]:
                        continue
                    if any(alias in p for alias in primary_aliases[col_idx]) and (
                        any(solvent in p for solvent in self.SOLVENT_KEYWORDS) or any(k in p for k in self.CAP_KEYWORDS)
                    ) and not (other_aliases and any(other in p for other in other_aliases)):
                        idxs.add(i)

            evidence = [paragraphs[i] for i in sorted(idxs)]
            per_column.append(evidence)

        return per_column, tech_paras

    def _score_paragraph(self, text: str) -> int:
        if not text:
            return 0
        t = normalize_for_parsing(text).lower()
        score = 0
        if any(k in t for k in self.SECTION_HINTS):
            score += 1
        if any(k in t for k in self.CAP_KEYWORDS):
            score += 2
        if any(k in t for k in self.SOLVENT_KEYWORDS):
            score += 2
        if "column" in t or "columns" in t:
            score += 2
        if re.search(r"\b\d+(?:\.\d+)?\s*(wt%|w/w|v/v|vol%|%)\b", t):
            score += 2
        if re.search(r"\b\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?\s*(vol%|v/v|%)", t):
            score += 2
        if re.search(r"\b\d+(?:\.\d+)?\s*(ml|min|mL|min|uL|ul)\b", t):
            score += 1
        if re.search(r"\b\d+(?:\.\d+)?\s*C\b", t):
            score += 1
        return score

    def _build_global_evidence(self, paragraphs: List[str]) -> str:
        scored: List[Tuple[int, int, str]] = []
        for i, p in enumerate(paragraphs):
            score = self._score_paragraph(p)
            if score <= 0:
                continue
            scored.append((score, i, p))
        scored.sort(key=lambda x: (-x[0], x[1]))

        selected: List[str] = []
        total_chars = 0
        for score, idx, p in scored:
            if total_chars >= self.max_global_chars:
                break
            selected.append(p)
            total_chars += len(p) + 2

        # fallback: if nothing scored, take first few paragraphs
        if not selected and paragraphs:
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
            for num in re.findall(r"\d+(?:\.\d+)?", str(ratio)):
                tokens.append(num)

        return list(dict.fromkeys(tokens))

    def _build_condition_evidence(self, cond: Dict[str, Any], paragraphs: List[str], max_chars: int) -> str:
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
        for score, idx, p in scored:
            if total_chars >= max_chars:
                break
            selected.append(p)
            total_chars += len(p) + 2

        return self._sanitize_for_llm("\n\n".join(selected))

    def _build_refine_prompt(self, condition: Dict[str, Any], evidence_text: str) -> str:
        schema = (
            '{"LCCC_conditions":[{"critical_polymer_unit":null,'
            '"SP_FIELDS":{"column_name":null,"material":null,"modification":null,"pore_size":null,'
            '"particle_diameter":null,"column_dimensions":null,"number_of_columns":null,"manufacturer":null,"phase":null},'
            '"SOL_FIELDS":{"solvent":null,"ratio":null,"ratio_units":null},'
            '"TECH_FIELDS":{"temperature":null,"flow_rate":null,"injected_polymer_concentration":null,'
            '"injected_polymer_solvent_solution":null,"injection_volume":null},'
            '"separation_behavior":{"critical_block_behavior":null,"non_critical_block_behavior":null,"purpose":null}}]}'
        )
        prompt = (
            "You refine an existing LCCC condition using evidence.\n"
            "Return ONLY valid JSON.\n"
            f"JSON schema:\n{schema}\n"
            "Rules:\n"
            "- Only use evidence text below.\n"
            "- Keep existing values if supported; fill missing if found.\n"
            "- Use null for unknown fields.\n"
            f"Current condition:\n{condition}\n"
            f"Evidence text:\n{evidence_text}\n"
        )
        return prompt

    def _build_evidence_text(self, evidence: List[str], tech_paras: List[str], max_chars: int) -> str:
        combined: List[str] = []
        seen: Set[str] = set()
        scored: List[Tuple[int, int, str, str]] = []
        for idx, p in enumerate(evidence):
            norm = normalize_for_parsing(p)
            if not norm or norm in seen:
                continue
            norm_l = norm.lower()
            score = 0
            if any(k in norm_l for k in self.CAP_KEYWORDS):
                score += 2
            if "acetone" in norm_l:
                score += 1
            scored.append((score, idx, p, norm))
            seen.add(norm)

        scored.sort(key=lambda item: (-item[0], item[1]))
        for _, _, p, _ in scored:
            combined.append(p)

        for p in tech_paras:
            norm = normalize_for_parsing(p)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            combined.append(p)

        text = "\n\n".join(combined)
        if len(text) > max_chars:
            text = text[:max_chars]
        return self._sanitize_for_llm(text)

    def _build_column_prompt(self, column: Dict[str, Any], evidence_text: str) -> str:
        schema = (
            '{"LCCC_conditions":[{"critical_polymer_unit":null,'
            '"SP_FIELDS":{"column_name":null,"material":null,"modification":null,"pore_size":null,'
            '"particle_diameter":null,"column_dimensions":null,"number_of_columns":null,"manufacturer":null,"phase":null},'
            '"SOL_FIELDS":{"solvent":null,"ratio":null,"ratio_units":null},'
            '"TECH_FIELDS":{"temperature":null,"flow_rate":null,"injected_polymer_concentration":null,'
            '"injected_polymer_solvent_solution":null,"injection_volume":null},'
            '"separation_behavior":{"critical_block_behavior":null,"non_critical_block_behavior":null,"purpose":null}}]}'
        )

        lines: List[str] = []
        for f in SP_FIELDS:
            v = column.get(f)
            if v:
                lines.append(f"{f}: {v}")
        col_spec = "\n".join(lines) if lines else "None"

        prompt = (
            "Extract LCCC (Liquid Chromatography at Critical Conditions) experimental data for this column.\n"
            "Return ONLY valid JSON.\n"
            f"JSON schema:\n{schema}\n"
            "\n"
            "CRITICAL UNDERSTANDING:\n"
            "- LCCC uses 'critical conditions' where one polymer block becomes 'chromatographically invisible'\n"
            "- Typically uses mixed solvents (acetone/water, ACN/water, methanol/water) with precise ratios\n"
            "- Different from SEC which uses single solvents (THF, chloroform)\n"
            "- Look for terms: 'critical condition', 'CAP', 'chromatographically invisible'\n"
            "\n"
            "Rules:\n"
            "- Output ONE condition if LCCC/critical conditions are described\n"
            "- Ignore SEC measurements unless combined with LCCC\n"
            "- critical_polymer_unit: the polymer at critical condition (e.g., 'PEO', 'PLLA', 'PS')\n"
            "- If multiple ratios given, use range format: '60-68' or '47/53'\n"
            "- ratio_units: typically 'v/v', 'wt%', or '%'\n"
            "- Use null for unknown fields\n"
            "- Solvent format: 'A / B' for mixtures (e.g., 'ACN / Water')\n"
            "\n"
            "Column specification:\n"
            f"{col_spec}\n"
            "\n"
            "Evidence text:\n"
            f"{evidence_text}\n"
        )
        return prompt

    def _build_prompt(self, chunk: LLMChunk) -> str:
        schema = (
            '{"LCCC_conditions":[{"critical_polymer_unit":null,'
            '"SP_FIELDS":{"column_name":null,"material":null,"modification":null,"pore_size":null,'
            '"particle_diameter":null,"column_dimensions":null,"number_of_columns":null,"manufacturer":null,"phase":null},'
            '"SOL_FIELDS":{"solvent":null,"ratio":null,"ratio_units":null},'
            '"TECH_FIELDS":{"temperature":null,"flow_rate":null,"injected_polymer_concentration":null,'
            '"injected_polymer_solvent_solution":null,"injection_volume":null},'
            '"separation_behavior":{"critical_block_behavior":null,"non_critical_block_behavior":null,"purpose":null}}]}'
        )

        prompt = (
            "You extract LCCC (liquid chromatography at critical conditions) conditions.\n"
            "Return ONLY valid JSON.\n"
            f"JSON schema:\n{schema}\n"
            "Rules:\n"
            "- Each condition corresponds to a distinct critical polymer unit plus a column + mobile phase.\n"
            "- Prefer conditions tied to CAP/critical conditions, not general theory.\n"
            "- Only include values explicitly stated in the text.\n"
            "- Use null for unknown fields.\n"
            "- Ignore SEC/GPC conditions unless explicitly tied to LCCC critical conditions.\n"
            '- If no conditions are described, return {"LCCC_conditions":[]}.\n'
            f"Text:\n{chunk.llm_text}\n"
        )
        return prompt

    def _build_global_prompt(self, evidence_text: str) -> str:
        schema = (
            '{"LCCC_conditions":[{"critical_polymer_unit":null,'
            '"SP_FIELDS":{"column_name":null,"material":null,"modification":null,"pore_size":null,'
            '"particle_diameter":null,"column_dimensions":null,"number_of_columns":null,"manufacturer":null,"phase":null},'
            '"SOL_FIELDS":{"solvent":null,"ratio":null,"ratio_units":null},'
            '"TECH_FIELDS":{"temperature":null,"flow_rate":null,"injected_polymer_concentration":null,'
            '"injected_polymer_solvent_solution":null,"injection_volume":null},'
            '"separation_behavior":{"critical_block_behavior":null,"non_critical_block_behavior":null,"purpose":null}}]}'
        )
        prompt = (
            "You extract LCCC conditions from the paper.\n"
            "Return ONLY valid JSON.\n"
            f"JSON schema:\n{schema}\n"
            "Rules:\n"
            "- Conditions may involve any critical unit (e.g., EO, PO, 1,4-PI).\n"
            "- Include only conditions explicitly stated in the text.\n"
            "- Use null for unknown fields.\n"
            "- Capture solvent mixtures and ratios (wt%, vol%, v/v).\n"
            "- Ignore purely theoretical discussion unless tied to experimental conditions.\n"
            f"Text:\n{evidence_text}\n"
        )
        return prompt

    def _build_prompt_with_catalog(self, chunk: LLMChunk, catalog: List[Dict[str, Any]]) -> str:
        prompt = self._build_prompt(chunk)
        if not catalog:
            return prompt
        lines: List[str] = []
        for col in catalog[:6]:
            parts = []
            for f in SP_FIELDS:
                v = col.get(f)
                if v:
                    parts.append(f"{f}={v}")
            if parts:
                lines.append("- " + "; ".join(parts))
        if lines:
            prompt = (
                prompt
                + "Known columns (from the paper; use only if supported by the text):\n"
                + "\n".join(lines)
                + "\n"
            )
        return prompt

    def _extract_columns_catalog(self, paragraphs: List[str], full_text: str) -> List[Dict[str, Any]]:
        """Extract column catalog using LLM - works for ANY column"""
        if not paragraphs:
            return []
        
        col_paras: List[str] = []
        for i, p in enumerate(paragraphs):
            if re.search(r"\bfollowing\s+columns\b|\bcolumns\s+were\s+used\b", p, re.I):
                col_paras.append(p)
                if i + 1 < len(paragraphs):
                    col_paras.append(paragraphs[i + 1])
        
        if not col_paras:
            col_paras = [p for p in paragraphs if "column" in (p or "").lower()]
        if not col_paras:
            return []
        
        text = "\n".join(col_paras)
        if len(text) > 4000:
            text = text[:4000]

        schema = (
            '{"columns":[{"column_name":null,"material":null,"modification":null,"pore_size":null,'
            '"particle_diameter":null,"column_dimensions":null,"number_of_columns":null,"manufacturer":null,"phase":null}]}'
        )
        prompt = (
            "Extract ALL column specifications from the text.\n"
            "Return ONLY valid JSON.\n"
            f"JSON schema:\n{schema}\n"
            "Rules:\n"
            "- Extract EVERY column mentioned\n"
            "- Only include values explicitly stated\n"
            "- Use null for unknown fields\n"
            f"Text:\n{self._sanitize_for_llm(text)}\n"
        )
        
        outputs = self.client.run_json(prompt)
        if not outputs:
            return []

        def _col_key(col: Dict[str, Any]) -> str:
            name = normalize_for_parsing(str(col.get("column_name") or "")).lower()
            if name:
                return f"name:{name}"
            manuf = normalize_for_parsing(str(col.get("manufacturer") or "")).lower()
            dims = normalize_for_parsing(str(col.get("column_dimensions") or "")).lower()
            phase = normalize_for_parsing(str(col.get("phase") or "")).lower()
            material = normalize_for_parsing(str(col.get("material") or "")).lower()
            return f"m:{manuf}|d:{dims}|p:{phase}|mat:{material}"

        def _score(col: Dict[str, Any]) -> int:
            return sum(1 for v in col.values() if v is not None)

        per_model: List[Tuple[str, List[Dict[str, Any]]]] = []
        for model_name, data in outputs:
            cols = []
            if isinstance(data, dict) and "columns" in data:
                cols = data.get("columns") or []
            elif isinstance(data, list):
                cols = data
            if not isinstance(cols, list):
                continue

            cleaned: List[Dict[str, Any]] = []
            for col in cols:
                if not isinstance(col, dict):
                    continue
                col_clean = {f: col.get(f) for f in SP_FIELDS}
                col_clean = self._filter_fields(
                    col_clean,
                    SP_FIELDS,
                    full_text,
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

        merged: List[Dict[str, Any]] = []
        for key, cols in candidates.items():
            if len(support.get(key, set())) < self.consensus_min:
                continue
            best = max(cols, key=_score)
            merged.append(best)

        return merged

    def _apply_catalog(self, sp_clean: Dict[str, Any], catalog: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not catalog:
            return sp_clean

        def _match(sp: Dict[str, Any], col: Dict[str, Any]) -> bool:
            for f in ("column_name", "phase", "modification", "material"):
                a = sp.get(f)
                b = col.get(f)
                if not a or not b:
                    continue
                a_norm = normalize_for_parsing(str(a)).lower()
                b_norm = normalize_for_parsing(str(b)).lower()
                if a_norm and b_norm and (a_norm in b_norm or b_norm in a_norm):
                    return True
            return False

        for col in catalog:
            if _match(sp_clean, col):
                for f in SP_FIELDS:
                    if sp_clean.get(f) is None and col.get(f) is not None:
                        sp_clean[f] = col.get(f)
                break
        return sp_clean

    def _matches_catalog(self, sp_clean: Dict[str, Any], catalog: List[Dict[str, Any]]) -> bool:
        if not catalog:
            return True

        def _match(sp: Dict[str, Any], col: Dict[str, Any]) -> bool:
            for f in ("column_name", "phase", "modification", "material"):
                a = sp.get(f)
                b = col.get(f)
                if not a or not b:
                    continue
                a_norm = normalize_for_parsing(str(a)).lower()
                b_norm = normalize_for_parsing(str(b)).lower()
                if a_norm and b_norm and (a_norm in b_norm or b_norm in a_norm):
                    return True
            return False

        return any(_match(sp_clean, col) for col in catalog)

    def _filter_fields(
        self,
        data: Dict[str, Any],
        fields: List[str],
        text: str,
        soft_fields: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        soft_fields = soft_fields or set()
        for f in fields:
            if f not in data:
                continue
            v = data.get(f)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                raw = int(v) if isinstance(v, float) and float(v).is_integer() else v
                v_str = str(raw)
            elif isinstance(v, str):
                raw = v.strip()
                v_str = raw
            else:
                data[f] = None
                continue

            if raw == "" or raw is None:
                data[f] = None
                continue

            if f in {"number_of_columns"}:
                data[f] = raw
                continue

            if value_in_text(v_str, text):
                data[f] = raw
                continue

            if f in soft_fields:
                if len(str(raw)) <= 80:
                    data[f] = raw
                else:
                    data[f] = None
                continue

            if f in {"ratio", "temperature", "flow_rate", "injection_volume"} and self._numbers_present(v_str, text):
                data[f] = raw
                continue

            if f in {"ratio_units"}:
                unit_tokens = ["wt%", "w/w", "v/v", "%"]
                t_norm = normalize_for_parsing(text).lower()
                if any(tok in t_norm for tok in unit_tokens):
                    data[f] = raw
                    continue

            data[f] = None
        return data

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

        unit_raw = cond.get("critical_polymer_unit") or cond.get("critical_unit")
        unit = None
        if isinstance(unit_raw, dict):
            # Take first key or value as a fallback label
            if unit_raw:
                first_key = next(iter(unit_raw.keys()))
                unit_raw = first_key or next(iter(unit_raw.values()))
        if isinstance(unit_raw, str):
            u = unit_raw.strip().upper()
            if "ETHYLENE" in u or u == "EO":
                unit = "EO"
            elif "PROPYLENE" in u or u == "PO":
                unit = "PO"
            else:
                unit = unit_raw.strip()

        sp_raw = cond.get("SP_FIELDS") or cond.get("stationary_phase") or {}
        sol_raw = cond.get("SOL_FIELDS") or cond.get("solvent_details") or {}
        tech_raw = cond.get("TECH_FIELDS") or cond.get("technical_details") or {}
        sep_raw = cond.get("separation_behavior") or cond.get("behavior") or {}

        if isinstance(sp_raw, list):
            sp_raw = sp_raw[0] if sp_raw else {}
        if isinstance(sol_raw, list):
            sol_raw = sol_raw[0] if sol_raw else {}
        if isinstance(tech_raw, list):
            tech_raw = tech_raw[0] if tech_raw else {}

        if not isinstance(sp_raw, dict):
            sp_raw = {}
        if not isinstance(sol_raw, dict):
            sol_raw = {}
        if not isinstance(tech_raw, dict):
            tech_raw = {}
        if not isinstance(sep_raw, dict):
            sep_raw = {}

        sp_clean = {f: sp_raw.get(f) for f in SP_FIELDS}
        sp_clean = self._filter_fields(
            sp_clean,
            SP_FIELDS,
            validation_text,
            soft_fields={"phase", "modification", "material", "manufacturer"},
        )
        if column_hint:
            for f in SP_FIELDS:
                if sp_clean.get(f) is None and column_hint.get(f) is not None:
                    sp_clean[f] = column_hint.get(f)

        sp_clean = self._apply_catalog(sp_clean, catalog)
        if not self._matches_catalog(sp_clean, catalog):
            return None

        sol_clean = {f: sol_raw.get(f) for f in SOL_FIELDS}
        sol_clean = self._filter_fields(sol_clean, SOL_FIELDS, validation_text)
        if not sol_clean.get("solvent"):
            t_norm = normalize_for_parsing(context_snippet or validation_text).lower()
            # Fallback: infer simple solvent pairs from context
            if "acetone" in t_norm and "water" in t_norm:
                sol_clean["solvent"] = "Acetone / Water"
            elif "butanone" in t_norm and "cyclohexane" in t_norm:
                sol_clean["solvent"] = "Butanone / Cyclohexane"
            elif "thf" in t_norm or "tetrahydrofuran" in t_norm:
                sol_clean["solvent"] = "THF"

        tech_clean = {f: tech_raw.get(f) for f in TECH_FIELDS}
        tech_clean = self._filter_fields(tech_clean, TECH_FIELDS, validation_text)

        sep_clean = {f: sep_raw.get(f) for f in SEP_FIELDS}
        sep_clean = {k: (v.strip() if isinstance(v, str) and v.strip() else None) for k, v in sep_clean.items()}

        has_data = (
            any(v is not None for v in sp_clean.values())
            or any(v is not None for v in sol_clean.values())
            or any(v is not None for v in tech_clean.values())
            or any(v is not None for v in sep_clean.values())
        )
        if not has_data:
            return None

        return {
            "critical_polymer_unit": unit,
            "SP_FIELDS": sp_clean,
            "SOL_FIELDS": sol_clean,
            "TECH_FIELDS": tech_clean,
            "separation_behavior": sep_clean,
            "context_snippet": context_snippet,
        }

    def _cond_key(self, cond: Dict[str, Any]) -> Tuple[str, str, str]:
        unit = normalize_for_parsing(str(cond.get("critical_polymer_unit") or "")).lower()
        sp = cond.get("SP_FIELDS") or {}
        sol = cond.get("SOL_FIELDS") or {}
        col = normalize_for_parsing(str(sp.get("column_name") or "")).lower()
        solvent = normalize_for_parsing(str(sol.get("solvent") or "")).lower()
        return (unit, col, solvent)

    def _apply_consensus(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.consensus_min <= 1:
            for c in conditions:
                c.pop("_model", None)
            return conditions

        support: Dict[Tuple[str, str, str], set] = {}
        for cond in conditions:
            key = self._cond_key(cond)
            model = cond.get("_model") or ""
            if not model:
                continue
            support.setdefault(key, set()).add(model)

        filtered: List[Dict[str, Any]] = []
        for cond in conditions:
            key = self._cond_key(cond)
            if len(support.get(key, set())) >= self.consensus_min:
                cond.pop("_model", None)
                filtered.append(cond)

        return filtered

    def _match_catalog_by_name(
        self, name: Optional[str], catalog: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not name or not catalog:
            return None
        name_norm = normalize_for_parsing(str(name)).lower()
        if not name_norm:
            return None
        for col in catalog:
            col_name = normalize_for_parsing(str(col.get("column_name") or "")).lower()
            if not col_name:
                continue
            if name_norm in col_name or col_name in name_norm:
                return col
        return None

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
        stats = {
            "llm_available": True,
            "chunks": 0,
            "llm_calls": 0,
            "conditions": 0,
            "parse_failures": 0,
            "mode": "dynamic",
            "columns_catalog": len(catalog),
        }

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
                        logger.info(
                            f"LLM column {idx + 1}/{len(catalog)} "
                            f"(evidence chars={len(evidence_text)})"
                        )

                    if not outputs:
                        continue

                    for model_name, data in outputs:
                        if not data:
                            stats["parse_failures"] += 1
                            continue

                        if isinstance(data, dict) and "LCCC_conditions" in data:
                            conditions = data.get("LCCC_conditions") or []
                        elif isinstance(data, dict) and "conditions" in data:
                            conditions = data.get("conditions") or []
                        elif isinstance(data, list):
                            conditions = data
                        else:
                            stats["parse_failures"] += 1
                            continue

                        if not isinstance(conditions, list):
                            stats["parse_failures"] += 1
                            continue

                        validation_text = evidence_text or full_text
                        context_snippet = normalize_for_parsing(evidence_text)[:1200]

                        for cond in conditions:
                            normalized = self._normalize_condition(
                                cond,
                                validation_text=validation_text,
                                catalog=catalog,
                                context_snippet=context_snippet,
                                column_hint=col,
                            )
                            if normalized:
                                normalized["_model"] = model_name
                                extracted.append(normalized)

                extracted = self._apply_consensus(extracted)
                logger.info(
                    f"LLM extraction produced {len(extracted)} conditions using per-column evidence"
                )

        # Global dynamic pass (high recall) for diverse papers
        if self.high_accuracy:
            global_text = self._build_global_evidence(paragraphs)
            if global_text:
                prompt = self._build_global_prompt(global_text)
                outputs = self.client.run_json(prompt)
                stats["llm_calls"] += max(1, len(self.models))
                if outputs:
                    for model_name, data in outputs:
                        if not data:
                            stats["parse_failures"] += 1
                            continue
                        if isinstance(data, dict) and "LCCC_conditions" in data:
                            conditions = data.get("LCCC_conditions") or []
                        elif isinstance(data, dict) and "conditions" in data:
                            conditions = data.get("conditions") or []
                        elif isinstance(data, list):
                            conditions = data
                        else:
                            stats["parse_failures"] += 1
                            continue

                        if not isinstance(conditions, list):
                            stats["parse_failures"] += 1
                            continue

                        validation_text = global_text or full_text
                        context_snippet = normalize_for_parsing(global_text)[:1200]

                        for cond in conditions:
                            normalized = self._normalize_condition(
                                cond,
                                validation_text=validation_text,
                                catalog=catalog,
                                context_snippet=context_snippet,
                            )
                            if normalized:
                                normalized["_model"] = model_name
                                extracted.append(normalized)

        chunks = self.build_chunks(paragraphs)
        if chunks:
            stats["chunks"] += len(chunks)
            for chunk in chunks:
                prompt = self._build_prompt_with_catalog(chunk, catalog)
                outputs = self.client.run_json(prompt)
                stats["llm_calls"] += max(1, len(self.models))
                if self.debug:
                    logger.info(
                        f"LLM chunk {chunk.chunk_id}/{len(chunks)} paras {chunk.para_start}-{chunk.para_end} "
                        f"(chars raw={len(chunk.raw_text)}, llm={len(chunk.llm_text)})"
                    )

                if not outputs:
                    continue

                for model_name, data in outputs:
                    if not data:
                        stats["parse_failures"] += 1
                        continue

                    if isinstance(data, dict) and "LCCC_conditions" in data:
                        conditions = data.get("LCCC_conditions") or []
                    elif isinstance(data, dict) and "conditions" in data:
                        conditions = data.get("conditions") or []
                    elif isinstance(data, list):
                        conditions = data
                    else:
                        stats["parse_failures"] += 1
                        continue

                    if not isinstance(conditions, list):
                        stats["parse_failures"] += 1
                        continue

                    validation_text = full_text or chunk.raw_text
                    context_snippet = normalize_for_parsing(chunk.raw_text)[:1200]

                    for cond in conditions:
                        normalized = self._normalize_condition(
                            cond,
                            validation_text=validation_text,
                            catalog=catalog,
                            context_snippet=context_snippet,
                        )
                        if normalized:
                            normalized["_model"] = model_name
                            extracted.append(normalized)

        extracted = self._apply_consensus(extracted)

        # Optional refinement pass for higher accuracy
        if self.high_accuracy and self.refine_enabled and extracted:
            refined: List[Dict[str, Any]] = []
            for cond in extracted[: self.refine_limit]:
                sp = cond.get("SP_FIELDS") or {}
                sol = cond.get("SOL_FIELDS") or {}
                tech = cond.get("TECH_FIELDS") or {}
                needs_refine = any(v is None for v in sp.values()) or any(v is None for v in sol.values()) or any(
                    v is None for v in tech.values()
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
                for model_name, data in outputs:
                    if not data:
                        continue
                    if isinstance(data, dict) and "LCCC_conditions" in data:
                        conditions = data.get("LCCC_conditions") or []
                    elif isinstance(data, dict) and "conditions" in data:
                        conditions = data.get("conditions") or []
                    elif isinstance(data, list):
                        conditions = data
                    else:
                        continue
                    if not isinstance(conditions, list) or not conditions:
                        continue

                    column_hint = self._match_catalog_by_name(sp.get("column_name"), catalog)
                    normalized = self._normalize_condition(
                        conditions[0],
                        validation_text=evidence_text,
                        catalog=catalog,
                        context_snippet=normalize_for_parsing(evidence_text)[:1200],
                        column_hint=column_hint,
                    )
                    if normalized:
                        updated = normalized
                        break

                refined.append(updated or cond)

            # Keep any extra conditions beyond refine limit
            if len(extracted) > self.refine_limit:
                refined.extend(extracted[self.refine_limit :])
            extracted = refined

        stats["conditions"] = len(extracted)
        stats["consensus_min"] = self.consensus_min
        stats["models"] = list(self.models)
        logger.info(f"LLM extraction produced {len(extracted)} conditions (dynamic)")
        return extracted, stats