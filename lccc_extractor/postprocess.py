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
        "\n"
        "RULES (apply in this order):\n"
        "\n"
        "1. FIX RATIO ERRORS (most important):\n"
        "   - ratio must always be the FULL mixture composition as 'A/B', e.g., '92/8'.\n"
        "   - If ratio is a single number like '8' or '92' for a binary solvent, convert it:\n"
        "     * If value < 50: it's the minor component → ratio = '(100-val)/val' e.g., '8' → '92/8'\n"
        "     * If value >= 50: it's the major component → ratio = 'val/(100-val)' e.g., '92' → '92/8'\n"
        "   - ratio '70' with solvent 'Butanone / Cyclohexane' → '70/30'\n"
        "   - ratio '97/3' with no ratio_units → add ratio_units='vol%'\n"
        "\n"
        "2. FIX COLUMN NAME ERRORS:\n"
        "   - Remove entries where column_name is a pore size (e.g., '10 5 A', '10^5 A', '100 A').\n"
        "     Move such values to pore_size column instead.\n"
        "   - Remove 'precolumn', 'guard column', 'SDV 5 um precolumn' entries — these are not analytical columns.\n"
        "   - Fix OCR errors: 'Nucleosol' → 'Nucleosil', 'Of C18' → 'C18', 'RP-18' in column_name → keep as-is but normalize phase='reversed phase'.\n"
        "\n"
        "3. MERGE DUPLICATES:\n"
        "   - Merge rows where polymer_group + manufacturer + phase + solvent + ratio are essentially the same.\n"
        "   - When merging, prefer non-empty values. Combine purpose text uniquely (join with '; ').\n"
        "   - Two ratios '92' and '92/8' for the same column/solvent → merge as '92/8'.\n"
        "\n"
        "4. NORMALIZE UNITS:\n"
        "   - temperature: always '29 °C' format (number + space + °C).\n"
        "   - flow_rate: '0.5 mL/min' format.\n"
        "   - injection_volume: '100 µL' or '100 uL' format.\n"
        "   - ratio_units: use 'vol%' or 'wt%' (not 'v/v', 'w/w', '%').\n"
        "   - injected_polymer_concentration: '2.7 mg/mL' format.\n"
        "\n"
        "5. DO NOT invent values. Only use what is in the input data.\n"
        "\n"
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