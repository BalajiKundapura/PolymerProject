from __future__ import annotations

from typing import Any, Dict


def prune_output(linked: Dict[str, Dict[str, Any]], mode: str = "compact") -> Dict[str, Dict[str, Any]]:
    """
    Remove debugging/evidence fields for a cleaner output schema.
    mode:
      - full: keep everything
      - compact: remove evidence/context/debug fields
    """
    if not mode or mode.lower() == "full":
        return linked

    drop_keys = {
        "context_snippet",
        "paragraph_indices",
        "evidence",
        "polymer_mentions",
        "completion",
    }

    pruned: Dict[str, Dict[str, Any]] = {}
    for poly, exps in linked.items():
        pruned[poly] = {}
        for exp_id, exp in exps.items():
            cleaned = {k: v for k, v in exp.items() if k not in drop_keys}
            pruned[poly][exp_id] = cleaned
    return pruned
