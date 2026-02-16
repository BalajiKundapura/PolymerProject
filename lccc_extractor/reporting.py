from __future__ import annotations

import csv
from typing import Any, Dict, Iterable, List

from .text.normalization import normalize_for_parsing


CONDITION_TABLE_COLUMNS = [
    "polymer_group",
    "critical_polymer_unit",
    "column_name",
    "manufacturer",
    "phase",
    "material",
    "modification",
    "column_dimensions",
    "number_of_columns",
    "pore_size",
    "particle_diameter",
    "solvent",
    "ratio",
    "ratio_units",
    "temperature",
    "flow_rate",
    "injection_volume",
    "injected_polymer_concentration",
    "injected_polymer_solvent_solution",
    "purpose",
]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:g}"
    return str(value)


def build_conditions_table(conditions: List[Dict[str, Any]], main_polymers: Iterable[str]) -> List[Dict[str, Any]]:
    polymer_fallback = ", ".join([p for p in main_polymers if p])
    if not polymer_fallback:
        polymer_fallback = "unknown"

    rows: List[Dict[str, Any]] = []
    for cond in conditions:
        sp = cond.get("SP_FIELDS") or {}
        sol = cond.get("SOL_FIELDS") or {}
        tech = cond.get("TECH_FIELDS") or {}
        sep = cond.get("separation_behavior") or {}

        polymer_group = cond.get("critical_polymer_unit") or polymer_fallback

        row = {
            "polymer_group": polymer_group,
            "critical_polymer_unit": _stringify(cond.get("critical_polymer_unit")),
            "column_name": _stringify(sp.get("column_name")),
            "manufacturer": _stringify(sp.get("manufacturer")),
            "phase": _stringify(sp.get("phase")),
            "material": _stringify(sp.get("material")),
            "modification": _stringify(sp.get("modification")),
            "column_dimensions": _stringify(sp.get("column_dimensions")),
            "number_of_columns": _stringify(sp.get("number_of_columns")),
            "pore_size": _stringify(sp.get("pore_size")),
            "particle_diameter": _stringify(sp.get("particle_diameter")),
            "solvent": _stringify(sol.get("solvent")),
            "ratio": _stringify(sol.get("ratio")),
            "ratio_units": _stringify(sol.get("ratio_units")),
            "temperature": _stringify(tech.get("temperature")),
            "flow_rate": _stringify(tech.get("flow_rate")),
            "injection_volume": _stringify(tech.get("injection_volume")),
            "injected_polymer_concentration": _stringify(tech.get("injected_polymer_concentration")),
            "injected_polymer_solvent_solution": _stringify(tech.get("injected_polymer_solvent_solution")),
            "purpose": _stringify(sep.get("purpose")),
        }
        rows.append(row)

    # Dedupe rows: collapse identical polymer + column + phase (+ solvent/ratio when present).
    merged: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("polymer_group") or "",
            row.get("critical_polymer_unit") or "",
            row.get("column_name") or "",
            row.get("phase") or "",
            row.get("solvent") or "",
            row.get("ratio") or "",
            row.get("ratio_units") or "",
            row.get("column_dimensions") or "",
            row.get("manufacturer") or "",
        )
        if key not in merged:
            merged[key] = row
            continue

        existing = merged[key]
        for field, value in row.items():
            if not existing.get(field) and value:
                existing[field] = value
        # Merge purpose if different
        if row.get("purpose") and row.get("purpose") not in (existing.get("purpose") or ""):
            if existing.get("purpose"):
                existing["purpose"] = f"{existing['purpose']}; {row['purpose']}"
            else:
                existing["purpose"] = row.get("purpose")

    return list(merged.values())


def build_polymers_table(polymer_counts: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, count in sorted(polymer_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        rows.append({"polymer": name, "count": count})
    return rows


def write_conditions_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        # still write header so downstream knows the schema
        rows = [{col: "" for col in CONDITION_TABLE_COLUMNS}]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CONDITION_TABLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
