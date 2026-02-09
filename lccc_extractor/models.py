from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class PolymerMention:
    """Represents a single mention of a polymer in text."""

    polymer_name: str
    canonical_name: str
    position: int  # character position in document text
    context_before: str  # up to ~200 chars before mention
    context_after: str  # up to ~200 chars after mention
    full_context: str  # full paragraph text
    para_idx: int

    def get_context_window(self) -> str:
        return f"{self.context_before} [{self.polymer_name}] {self.context_after}"


@dataclass
class StationaryPhase:
    column_name: Optional[str] = None
    material: Optional[str] = None
    modification: Optional[str] = None
    pore_size: Optional[str] = None
    particle_diameter: Optional[str] = None
    column_dimensions: Optional[str] = None
    number_of_columns: Optional[str] = None
    manufacturer: Optional[str] = None
    phase: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v if v not in ("", None) else None) for k, v in asdict(self).items()}


@dataclass
class SolventDetail:
    solvent: Optional[str] = None
    ratio: Optional[str] = None
    ratio_units: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v if v not in ("", None) else None) for k, v in asdict(self).items()}


@dataclass
class TechnicalDetail:
    temperature: Optional[str] = None
    flow_rate: Optional[str] = None
    injected_polymer_concentration: Optional[str] = None
    injected_polymer_solvent_solution: Optional[str] = None
    injection_volume: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v if v not in ("", None) else None) for k, v in asdict(self).items()}

