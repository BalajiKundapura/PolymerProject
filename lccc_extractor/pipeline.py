import json
import os
import time
from typing import Any, Dict, Optional, Set, Tuple

from polymerSubject import extract_polymers, select_main_polymers, validate_polymers

from .classifier import get_classifier
from .completion import complete_linked_records
from .context import (
    classify_paragraphs,
    extract_entities_context_aware,
    extract_global_context,
    link_context,
    validate_output,
)
from .llm_fallback import LLMConfig, LLMFallback
from .logging_config import logger
from .output import prune_output
from .polymer_mentions import find_polymer_mentions
from .text_utils import clean_text


def run_pipeline(
    text: str,
    use_validation: bool = True,
    threshold_ratio: float = 0.5,
    completion_mode: str = "balanced",
    llm_config: Optional[LLMConfig] = None,
    output_mode: str = "compact",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Run context-aware LCCC extraction pipeline.
    Returns (linked_experiments_by_polymer, metadata).
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

    known_polymers_set: Set[str] = set(main_polymers.keys())

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
    clf = get_classifier()
    if any(getattr(global_tech, k) is not None for k in global_tech.__dataclass_fields__):
        logger.info(f"  Global technical: {global_tech.to_dict()}")

    # Step 5: Find polymer mentions with context
    logger.info("Step 5: Finding polymer mentions with context windows...")
    mentions = find_polymer_mentions(text, paragraphs, known_polymers_set)
    logger.info(f"  Found {len(mentions)} polymer mentions")

    # Step 6: Extract LCCC entities using context
    logger.info("Step 6: Extracting LCCC experimental details from context...")
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

    # Step 7b: Completion pass (optional)
    logger.info("Step 7b: Completing records...")
    llm = None
    if completion_mode and completion_mode.lower() == "aggressive":
        llm = LLMFallback(llm_config)
        if llm and not llm.available:
            logger.warning("LLM fallback requested but not available (ollama not found). Proceeding without LLM.")
            llm = None
    linked, completion_stats = complete_linked_records(linked, completion_mode=completion_mode, llm=llm)

    # Step 7c: Prune output for a clean schema
    linked = prune_output(linked, mode=output_mode)

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
        "main_polymers_list": list(main_polymers.keys()),
        "completion": completion_stats,
        "output_mode": output_mode,
    }

    return linked, metadata


def save_json(data: Dict[str, Dict[str, Any]], out_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save structured JSON to file with optional metadata."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    output = {"metadata": metadata or {}, "experiments": data}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f" Saved structured JSON to {out_path}")
