import json
import re
from typing import Any, Dict, List, Set, Tuple

from .classifier import ParagraphClassifier, get_classifier
from .extractors import (
    _dedup_stationary_phases,
    _extract_solvents,
    _extract_stationary_phases,
    _extract_technical,
    _has_stationary_phase_data,
    _has_technical_data,
    _match_stationary_phases,
    _merge_technical,
    _merge_technical_fill,
)
from .logging_config import logger
from .models import PolymerMention, SolventDetail, StationaryPhase, TechnicalDetail
from .normalization import normalize_for_parsing
from .patterns import RE_DIM, RE_MANUF_LINE, RE_PARTICLE, RE_PORE


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

        # Avoid pulling SEC/GPC context into LCCC conditions.
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


def extract_entities_context_aware(
    mentions: List[PolymerMention],
    labeled_paragraphs: List[Tuple[str, str]],
    clf: ParagraphClassifier,
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

    def _solvent_key(name: str) -> str:
        if not name:
            return ""
        key = normalize_for_parsing(name).lower()
        key = re.sub(r"\([^)]*\)", "", key)
        key = re.sub(r"\s+", "", key)
        key = key.replace("-", "/")
        return key

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

        # Extraction window: use a wider context when it doesn't look SEC/GPC-related.
        # Mixed SEC/LCCC paragraphs are common in polymer papers; limiting the window helps prevent
        # pulling SEC conditions into LCCC experiments.
        ctx_has_sec = bool(re.search(r"\b(?:sec|gpc)\b|size exclusion|gel permeation", ctx_text.lower()))
        extract_text = para_text if ctx_has_sec else ctx_text

        local_cols = _extract_stationary_phases(extract_text)
        local_sols = _extract_solvents(extract_text)
        local_tech = _extract_technical(extract_text)

        matched_cols = _match_stationary_phases(para_text, global_columns)
        if not matched_cols:
            matched_cols = _match_stationary_phases(ctx_text, global_columns)
        def _tech_overrides_global() -> bool:
            if not _has_technical_data(local_tech):
                return False
            for k in local_tech.__dataclass_fields__:
                lv = getattr(local_tech, k)
                gv = getattr(global_tech, k)
                if lv is not None and gv is not None and lv != gv:
                    return True
            return False

        overrides_global = _tech_overrides_global()

        if overrides_global:
            # Treat as a separate condition block: do not fill from global defaults.
            cols = local_cols
            tech = local_tech
            sols = local_sols
        else:
            cols = matched_cols or local_cols
            if not cols and global_columns:
                # Fallback: if a paragraph has LCCC context but doesn't restate the column,
                # attach the global methods column(s) so experiment records are cohesive.
                cols = global_columns

            tech = _merge_technical(global_tech, local_tech)

            sols = local_sols
            if not sols and global_solvents_with_ratio:
                # Only attach global solvent ratios when the solvent name appears locally.
                extract_norm = normalize_for_parsing(extract_text).lower()
                extract_compact = re.sub(r"\s+", "", extract_norm).replace("-", "/")
                candidates = []
                for gs in global_solvents_with_ratio:
                    key = _solvent_key(gs.solvent or "")
                    if key and key in extract_compact:
                        candidates.append(gs)
                if candidates:
                    sols = candidates
            elif sols and global_solvents_with_ratio:
                # If we found the solvent/mixture name locally but the ratio only globally, merge them.
                local_keys = {_solvent_key(s.solvent or "") for s in sols if s.solvent}
                if local_keys:
                    for gs in global_solvents_with_ratio:
                        if _solvent_key(gs.solvent or "") in local_keys:
                            sols.append(gs)

        has_data = bool(cols) or bool(sols) or _has_technical_data(tech)
        if not has_data:
            continue

        for canonical, ms in by_para[para_idx].items():
            unique_mentions = sorted({m.polymer_name for m in ms})
            context_window = ms[0].get_context_window()[:700]

            items.append(
                {
                    "para_idx": para_idx,
                    "polymer_name": canonical,
                    "polymer_mentions": unique_mentions,
                    "context_window": context_window,
                    "is_lccc": True,
                    "stationary_phases": cols,
                    "solvent_details": sols,
                    "technical_details": tech,
                }
            )

            logger.info(f"Para {para_idx}: Extracted LCCC data for {canonical}")

    logger.info(f"Extracted {len(items)} LCCC experiments from {len(mentions)} polymer mentions")
    return items


def link_context(extracted: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build structured output per polymer."""
    def _count_non_null(d: Dict[str, Any]) -> int:
        return sum(1 for v in d.values() if v is not None)

    def _dict_subset(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        # True if all non-null fields in a are also present and equal in b.
        for k, v in a.items():
            if v is None:
                continue
            if b.get(k) != v:
                return False
        return True

    def _prune_redundant_stationary_phases(sp_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(sp_dicts) <= 1:
            return sp_dicts

        keep: List[Dict[str, Any]] = []
        for i, a in enumerate(sp_dicts):
            dominated = False
            for j, b in enumerate(sp_dicts):
                if i == j:
                    continue
                if _dict_subset(a, b) and _count_non_null(b) > _count_non_null(a):
                    dominated = True
                    break
            if not dominated:
                keep.append(a)

        keep.sort(key=_count_non_null, reverse=True)
        return keep

    def _prune_redundant_solvents(sol_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not sol_dicts:
            return sol_dicts

        # Dedup exact triples
        dedup: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for s in sol_dicts:
            key = (s.get("solvent") or "", s.get("ratio") or "", s.get("ratio_units") or "")
            dedup[key] = s
        sol_dicts = list(dedup.values())

        # If a ratio exists for the same solvent, drop the no-ratio placeholder entry.
        solvents_with_ratio = {s.get("solvent") for s in sol_dicts if s.get("solvent") and s.get("ratio")}
        filtered: List[Dict[str, Any]] = []
        for s in sol_dicts:
            if s.get("solvent") in solvents_with_ratio and not s.get("ratio"):
                continue
            filtered.append(s)

        # If units exist for the same solvent+ratio, drop the no-units variant.
        units_by_key = {(s.get("solvent") or "", s.get("ratio") or "") for s in filtered if s.get("solvent") and s.get("ratio") and s.get("ratio_units")}
        filtered2: List[Dict[str, Any]] = []
        for s in filtered:
            key = (s.get("solvent") or "", s.get("ratio") or "")
            if key in units_by_key and s.get("ratio") and not s.get("ratio_units"):
                continue
            filtered2.append(s)

        filtered2.sort(key=lambda s: ((s.get("solvent") or "").lower(), 0 if s.get("ratio") else 1, (s.get("ratio") or "")))
        return filtered2


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
        sp_dicts = _prune_redundant_stationary_phases(sp_dicts)

        sol_dicts = [s.to_dict() for s in sols]
        sol_dicts = _prune_redundant_solvents(sol_dicts)
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
    """Validate extracted data structure."""
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
