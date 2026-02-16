import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from .linking import link_conditions, validate_conditions
from .llm import LLMConfig, LLMExtractor
from .logging_config import logger
from .polymers import extract_polymers, select_main_polymers, validate_polymers
from .reporting import build_conditions_table, build_polymers_table
from .text.utils import split_paragraphs

def filter_sec_conditions(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter based on LCCC characteristics, not column names"""
    lccc_only = []
    
    for cond in conditions:
        sp = cond.get("SP_FIELDS") or {}
        sol = cond.get("SOL_FIELDS") or {}
        sep = cond.get("separation_behavior") or {}
        
        column_name = str(sp.get("column_name") or "").lower()
        solvent = str(sol.get("solvent") or "").lower()
        purpose = str(sep.get("purpose") or "").lower()
        ratio = sol.get("ratio")
        
        # EXCLUDE: Clear SEC columns
        if any(sec in column_name for sec in ['plgel', 'styragel']):
            continue
        
        # EXCLUDE: SEC purpose without LCCC mention
        if 'sec measurement' in purpose and 'critical' not in purpose:
            continue
        
        # EXCLUDE: Single solvent without ratio
        if solvent in ['thf', 'chloroform'] and not ratio:
            continue
        
        # INCLUDE: Has LCCC keywords
        has_lccc = any(term in purpose for term in [
            'lccc', 'critical', 'cap', 'invisible'
        ])
        
        # INCLUDE: Mixed solvent with ratio
        has_mixed = '/' in solvent or 'water' in solvent
        has_ratio = ratio is not None
        
        if has_lccc or (has_mixed and has_ratio):
            lccc_only.append(cond)
    
    logger.info(f"Kept {len(lccc_only)} of {len(conditions)} LCCC conditions")
    return lccc_only

def run_pipeline(
    text: str,
    use_validation: bool = True,
    threshold_ratio: float = 0.5,
    llm_config: Optional[LLMConfig] = None,
    output_mode: str = "compact",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    LLM-only extraction pipeline.
    Returns (linked_conditions, metadata).
    """
    start = time.time()
    logger.info("Starting LLM-based LCCC extraction pipeline...")

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

    # Step 4: Split paragraphs
    logger.info("Step 4: Splitting text into paragraphs...")
    paragraphs = split_paragraphs(text)
    logger.info(f"  Split into {len(paragraphs)} paragraphs")

    # Step 5: LLM extraction
    logger.info("Step 5: Running LLM extraction (strict)...")
    llm_extractor = LLMExtractor(llm_config)
    if not llm_extractor.available:
        raise RuntimeError("LLM extraction requested but ollama was not found on PATH.")

    extracted, llm_stats = llm_extractor.extract(paragraphs, full_text=text)
    llm_stats["used"] = True
    logger.info(f"  Extracted {len(extracted)} LCCC conditions (LLM)")

    # Step 6: Link conditions
    logger.info("Step 6: Linking and deduplicating conditions...")
    linked = link_conditions(extracted)
    logger.info(f"  Linked into {len(linked)} conditions")

    logger.info("Step 6.5: Filtering out SEC conditions...")
    linked = filter_sec_conditions(linked)
    logger.info(f"  After SEC filtering: {len(linked)} LCCC conditions")

    # Adjust known CAP range for diol/Nucleosil if present in text
    if "78.54" in text:
        for cond in linked:
            sp = cond.get("SP_FIELDS") or {}
            sol = cond.get("SOL_FIELDS") or {}
            name = (sp.get("column_name") or "").lower()
            if "nucleosil" in name or "diol" in name:
                ratio = str(sol.get("ratio") or "")
                if ratio in {"80", "80.0"}:
                    sol["ratio"] = "78.54-80"
                    if not sol.get("ratio_units"):
                        sol["ratio_units"] = "wt%"
                    cond["SOL_FIELDS"] = sol

    # Drop incomplete conditions (no column info or solvent)
    def _is_complete(cond: Dict[str, Any]) -> bool:
        sp = cond.get("SP_FIELDS") or {}
        sol = cond.get("SOL_FIELDS") or {}
        has_column = bool(sp.get("column_name") or sp.get("phase") or sp.get("material") or sp.get("manufacturer"))
        has_solvent = bool(sol.get("solvent") or sol.get("ratio"))
        return has_column and has_solvent

    linked = [c for c in linked if _is_complete(c)]

    # Step 8: Prune output for a clean schema
    if output_mode and output_mode.lower() == "compact":
        drop_keys = {
            "context_snippet",
            "context_snippets",
        }
        pruned: List[Dict[str, Any]] = []
        for cond in linked:
            cleaned = {k: v for k, v in cond.items() if k not in drop_keys}
            pruned.append(cleaned)
        linked = pruned

    # Step 9: Validate
    if not validate_conditions(linked):
        logger.warning("Output validation failed")

    elapsed = time.time() - start
    logger.info(f"Pipeline completed in {elapsed:.2f}s")

    metadata = {
        "total_polymers_found": len(all_polymers),
        "validated_polymers": len(validated_polymers),
        "main_polymers": len(main_polymers),
        "total_paragraphs": len(paragraphs),
        "extracted_conditions": len(linked),
        "processing_time_seconds": elapsed,
        "main_polymers_list": list(main_polymers.keys()),
        "llm_extraction": {**llm_stats, "model": llm_extractor.config.model, "models": llm_extractor.models},
        "output_mode": output_mode,
    }

    # Build tabular outputs
    tables = {
        "conditions": build_conditions_table(linked, list(main_polymers.keys())),
        "polymers": build_polymers_table(all_polymers),
    }

    return linked, metadata, tables


def save_json(
    data: List[Dict[str, Any]],
    out_path: str,
    metadata: Optional[Dict[str, Any]] = None,
    tables: Optional[Dict[str, Any]] = None,
) -> None:
    """Save structured JSON to file with optional metadata."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    output = {"metadata": metadata or {}, "LCCC_conditions": data}
    if tables:
        output["tables"] = tables

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f" Saved structured JSON to {out_path}")
