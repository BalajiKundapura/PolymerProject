from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .llm import LLMClient, LLMConfig
from .reporting import CONDITION_TABLE_COLUMNS


def llm_normalize_conditions_table(
    rows: List[Dict[str, Any]],
    llm_config: Optional[LLMConfig] = None,
    max_rows: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not rows:
        return rows, {"used": False, "reason": "empty"}

    client = LLMClient(llm_config)
    if not client.available:
        return rows, {"used": False, "reason": "llm_unavailable"}

    if len(rows) > max_rows:
        return rows, {"used": False, "reason": "row_limit", "row_count": len(rows)}

    schema = {"conditions": [dict.fromkeys(CONDITION_TABLE_COLUMNS, "")]}
    prompt = (
        "You clean and deduplicate a tabular LCCC conditions dataset.\n"
        "Return ONLY valid JSON with this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "Rules:\n"
        "- Keep the same column names and order.\n"
        "- Fix obvious OCR errors (e.g., Nucleosol->Nucleosil, 'Of C18'->'C18').\n"
        "- Normalize units (uL, mL/min, °C, wt%, vol%).\n"
        "- Merge duplicate rows if polymer_group + column_name + phase + solvent + ratio match.\n"
        "- When merging, prefer non-empty values and combine purpose text uniquely.\n"
        "- Do NOT invent values.\n"
        "Input JSON:\n"
        f"{json.dumps({'conditions': rows}, ensure_ascii=False)}\n"
    )

    outputs = client.run_json(prompt)
    if not outputs:
        return rows, {"used": False, "reason": "no_output"}

    # Use first successful model output
    for model_name, data in outputs:
        if isinstance(data, dict) and "conditions" in data and isinstance(data["conditions"], list):
            cleaned = data["conditions"]
            return cleaned, {"used": True, "model": model_name, "row_count": len(cleaned)}

    return rows, {"used": False, "reason": "invalid_output"}
