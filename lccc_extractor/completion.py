from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .extractors import _extract_technical, _has_technical_data
from .logging_config import logger
from .llm_fallback import LLMFallback
from .normalization import normalize_for_parsing

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


def _normalize_key(value: str) -> str:
    if not value:
        return ""
    key = normalize_for_parsing(value).lower()
    key = re.sub(r"\([^)]*\)", "", key)
    key = re.sub(r"[^a-z0-9]+", "", key)
    return key


def _solvent_key(value: str) -> str:
    if not value:
        return ""
    key = normalize_for_parsing(value).lower()
    key = re.sub(r"\([^)]*\)", "", key)
    key = re.sub(r"\s+", "", key)
    key = key.replace("-", "/")
    return key


def _value_in_text(value: str, text: str) -> bool:
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


def _count_non_null(d: Dict[str, Any]) -> int:
    return sum(1 for v in d.values() if v not in ("", None))


def _best_dict(existing: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> Dict[str, Any]:
    if existing is None:
        return candidate
    return candidate if _count_non_null(candidate) > _count_non_null(existing) else existing


def _missing_fields(exp: Dict[str, Any]) -> List[str]:
    missing: List[str] = []

    sp_list = exp.get("stationary_phase", []) or []
    if not sp_list:
        missing.append("stationary_phase")
    else:
        for f in SP_FIELDS:
            if all((sp.get(f) is None) for sp in sp_list):
                missing.append(f"stationary_phase.{f}")

    sol_list = exp.get("solvent_details", []) or []
    if not sol_list:
        missing.append("solvent_details")
    else:
        if any(s.get("ratio") is None for s in sol_list):
            missing.append("solvent_details.ratio")
        if any(s.get("ratio_units") is None for s in sol_list):
            missing.append("solvent_details.ratio_units")

    tech_list = exp.get("technical_details", []) or []
    tech = tech_list[0] if tech_list else {}
    for f in TECH_FIELDS:
        if tech.get(f) is None:
            missing.append(f"technical_details.{f}")

    return missing


def _merge_missing_fields(target: Dict[str, Any], source: Dict[str, Any], fields: List[str]) -> List[str]:
    filled: List[str] = []
    for f in fields:
        if target.get(f) is None and source.get(f) is not None:
            target[f] = source.get(f)
            filled.append(f)
    return filled


def _collect_context(exp: Dict[str, Any]) -> str:
    snippets: List[str] = []
    if exp.get("context_snippet"):
        snippets.append(exp["context_snippet"])
    for ev in exp.get("evidence", []) or []:
        if ev.get("context_snippet"):
            snippets.append(ev["context_snippet"])
    return " ".join(snippets).strip()


def complete_linked_records(
    linked: Dict[str, Dict[str, Any]],
    completion_mode: str = "balanced",
    llm: Optional[LLMFallback] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Best-effort completion pass to fill missing values without inventing data.
    completion_mode: strict | balanced | aggressive
    """
    completion_mode = (completion_mode or "balanced").lower()
    llm_available = bool(llm and llm.available)

    stats = {
        "completion_mode": completion_mode,
        "filled_fields": 0,
        "llm_calls": 0,
        "llm_filled_fields": 0,
    }

    # Build per-polymer indexes for safe filling.
    poly_col_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    poly_solvent_ratio: Dict[str, Dict[str, Set[Tuple[str, str]]]] = {}
    poly_unique_cols: Dict[str, Set[str]] = {}
    poly_unique_sols: Dict[str, Set[str]] = {}

    for poly, exps in linked.items():
        col_index: Dict[str, Dict[str, Any]] = {}
        sol_index: Dict[str, Set[Tuple[str, str]]] = {}
        unique_cols: Set[str] = set()
        unique_sols: Set[str] = set()

        for exp in exps.values():
            for sp in exp.get("stationary_phase", []) or []:
                name_key = _normalize_key(sp.get("column_name") or "")
                manuf_key = _normalize_key(sp.get("manufacturer") or "")
                dims_key = _normalize_key(sp.get("column_dimensions") or "")
                phase_key = _normalize_key(sp.get("phase") or "")

                if name_key:
                    key = f"name:{name_key}"
                    col_index[key] = _best_dict(col_index.get(key), sp)
                    unique_cols.add(key)
                if manuf_key:
                    key = f"manuf:{manuf_key}"
                    col_index[key] = _best_dict(col_index.get(key), sp)
                    unique_cols.add(key)
                if manuf_key and dims_key:
                    key = f"manufdims:{manuf_key}:{dims_key}"
                    col_index[key] = _best_dict(col_index.get(key), sp)
                    unique_cols.add(key)
                if phase_key:
                    key = f"phase:{phase_key}"
                    col_index[key] = _best_dict(col_index.get(key), sp)
                    unique_cols.add(key)

            for sol in exp.get("solvent_details", []) or []:
                sol_name = sol.get("solvent")
                ratio = sol.get("ratio")
                units = sol.get("ratio_units") or ""
                if not sol_name:
                    continue
                s_key = _solvent_key(sol_name)
                if not s_key:
                    continue
                unique_sols.add(s_key)
                if ratio:
                    sol_index.setdefault(s_key, set()).add((ratio, units))

        poly_col_index[poly] = col_index
        poly_solvent_ratio[poly] = sol_index
        poly_unique_cols[poly] = unique_cols
        poly_unique_sols[poly] = unique_sols

    # Completion pass per experiment
    for poly, exps in linked.items():
        col_index = poly_col_index.get(poly, {})
        sol_index = poly_solvent_ratio.get(poly, {})

        for exp_id, exp in exps.items():
            context = _collect_context(exp)
            filled_fields: List[str] = []

            sp_list = exp.get("stationary_phase", []) or []
            # Fill missing station phase fields using same column matches.
            for sp in sp_list:
                name_key = _normalize_key(sp.get("column_name") or "")
                manuf_key = _normalize_key(sp.get("manufacturer") or "")
                dims_key = _normalize_key(sp.get("column_dimensions") or "")
                phase_key = _normalize_key(sp.get("phase") or "")

                candidates: List[Dict[str, Any]] = []
                if name_key:
                    candidates.append(col_index.get(f"name:{name_key}"))
                if manuf_key and dims_key:
                    candidates.append(col_index.get(f"manufdims:{manuf_key}:{dims_key}"))
                if manuf_key:
                    candidates.append(col_index.get(f"manuf:{manuf_key}"))

                # In aggressive mode, allow phase-only matches if it's unique for this polymer.
                if completion_mode == "aggressive" and phase_key and len({k for k in poly_unique_cols.get(poly, set()) if k.startswith("phase:")}) == 1:
                    candidates.append(col_index.get(f"phase:{phase_key}"))

                for cand in candidates:
                    if cand:
                        filled_fields += _merge_missing_fields(sp, cand, SP_FIELDS)

            # Fill solvent ratios when unique for the solvent name.
            for sol in exp.get("solvent_details", []) or []:
                if not sol.get("solvent") or sol.get("ratio"):
                    continue
                key = _solvent_key(sol.get("solvent") or "")
                ratios = sol_index.get(key, set())
                if len(ratios) == 1:
                    ratio, units = next(iter(ratios))
                    if completion_mode == "aggressive" or _value_in_text(ratio, context):
                        sol["ratio"] = ratio
                        sol["ratio_units"] = units or sol.get("ratio_units")
                        filled_fields.append("solvent_details.ratio")

            # Fill technical fields from local context snippet.
            tech_list = exp.get("technical_details", []) or []
            tech = tech_list[0] if tech_list else {}
            extracted_td = _extract_technical(context) if context else None
            if extracted_td and _has_technical_data(extracted_td):
                extracted_dict = extracted_td.to_dict()
                filled_fields += _merge_missing_fields(tech, extracted_dict, TECH_FIELDS)
                if not tech_list:
                    exp["technical_details"] = [tech]
                else:
                    exp["technical_details"][0] = tech

            # Optional LLM completion for corner cases.
            missing_before = set(_missing_fields(exp))
            if llm_available and completion_mode == "aggressive" and missing_before:
                suggestion = llm.suggest(context)
                stats["llm_calls"] += 1
                if suggestion:
                    llm_filled: List[str] = []

                    sp_suggest = suggestion.get("stationary_phase") or {}
                    if sp_suggest:
                        # Ensure station phase list exists
                        if not exp.get("stationary_phase"):
                            exp["stationary_phase"] = [{}]
                        sp = exp["stationary_phase"][0]
                        for f in SP_FIELDS:
                            v = sp_suggest.get(f)
                            if sp.get(f) is None and v is not None and _value_in_text(v, context):
                                sp[f] = v
                                llm_filled.append(f"stationary_phase.{f}")

                    tech_suggest = suggestion.get("technical_details") or {}
                    if tech_suggest:
                        if not exp.get("technical_details"):
                            exp["technical_details"] = [{}]
                        tech = exp["technical_details"][0]
                        for f in TECH_FIELDS:
                            v = tech_suggest.get(f)
                            if tech.get(f) is None and v is not None and _value_in_text(v, context):
                                tech[f] = v
                                llm_filled.append(f"technical_details.{f}")

                    sol_suggest = suggestion.get("solvent_details") or []
                    if sol_suggest:
                        exp.setdefault("solvent_details", [])
                        for sol in sol_suggest:
                            solvent_name = sol.get("solvent")
                            if not solvent_name or not _value_in_text(solvent_name, context):
                                continue
                            key = _solvent_key(solvent_name)
                            existing = None
                            for ex_sol in exp["solvent_details"]:
                                if _solvent_key(ex_sol.get("solvent") or "") == key:
                                    existing = ex_sol
                                    break
                            if existing is None:
                                existing = {"solvent": solvent_name}
                                exp["solvent_details"].append(existing)
                                llm_filled.append("solvent_details.solvent")
                            for f in ("ratio", "ratio_units"):
                                v = sol.get(f)
                                if existing.get(f) is None and v is not None and _value_in_text(v, context):
                                    existing[f] = v
                                    llm_filled.append(f"solvent_details.{f}")

                    if llm_filled:
                        filled_fields.extend(llm_filled)
                        stats["llm_filled_fields"] += len(llm_filled)

            stats["filled_fields"] += len(filled_fields)
            if completion_mode != "strict":
                exp["completion"] = {
                    "filled_fields": sorted(set(filled_fields)),
                    "missing_fields": sorted(set(_missing_fields(exp))),
                    "llm_used": bool(llm_available and completion_mode == "aggressive"),
                }

    return linked, stats
