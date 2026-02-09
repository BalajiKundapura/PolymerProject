from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .logging_config import logger


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen3:8b"
    timeout_s: int = 30
    max_context_chars: int = 1600


class LLMFallback:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        if config is None:
            config = LLMConfig(
                provider=os.getenv("LCCC_LLM_PROVIDER", "ollama"),
                model=os.getenv("LCCC_LLM_MODEL", "qwen3:8b"),
                timeout_s=int(os.getenv("LCCC_LLM_TIMEOUT", "30")),
                max_context_chars=int(os.getenv("LCCC_LLM_MAX_CONTEXT", "1600")),
            )
        self.config = config
        self.available = self._is_available()

    def _is_available(self) -> bool:
        if self.config.provider.lower() == "ollama":
            return shutil.which("ollama") is not None
        return False

    def _run_ollama(self, prompt: str) -> Optional[str]:
        cmd = ["ollama", "run", self.config.model]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.config.timeout_s,
            )
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None
        if proc.returncode != 0:
            logger.warning(f"LLM call returned error: {proc.stderr.strip()}")
            return None
        return proc.stdout.strip()

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        # Strip common markdown wrappers.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
        m = re.search(r"\{.*\}", cleaned, re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def suggest(self, context: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            return None
        if not context:
            return None

        context = context[: self.config.max_context_chars]
        prompt = (
            "You are extracting LCCC experimental conditions from the text below.\n"
            "Return ONLY a JSON object with these keys:\n"
            "  - stationary_phase: object with keys column_name, material, modification, pore_size, particle_diameter,\n"
            "    column_dimensions, number_of_columns, manufacturer, phase\n"
            "  - solvent_details: list of objects with keys solvent, ratio, ratio_units\n"
            "  - technical_details: object with keys temperature, flow_rate, injected_polymer_concentration,\n"
            "    injected_polymer_solvent_solution, injection_volume\n"
            "Rules:\n"
            "  - Only include values that appear explicitly in the text.\n"
            "  - Use null for unknown fields.\n"
            "  - Do not invent values.\n"
            "Text:\n"
            f"{context}\n"
        )

        output = None
        if self.config.provider.lower() == "ollama":
            output = self._run_ollama(prompt)
        if not output:
            return None
        return self._extract_json(output)
