import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ..logging_config import logger


@dataclass
class LLMConfig:
    provider: str = "ollama"
    model: str = "qwen3:14b"
    models: List[str] = field(default_factory=list)
    timeout_s: int = 120                # raised from 60 — local models need headroom
    max_context_chars: int = 6000
    consensus_min: int = 0
    host: str = ""
    start_server: bool = True
    startup_timeout_s: int = 15
    use_http: bool = True
    timeout_chars_per_second: int = 20  # adaptive: add 1s per N chars of prompt
    max_timeout_s: int = 300            # hard ceiling on adaptive timeout
    retry_attempts: int = 2             # how many times to retry a timed-out call
    retry_delay_s: float = 2.0          # seconds to wait between retries


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        if config is None:
            config = LLMConfig(
                provider=os.getenv("LCCC_LLM_PROVIDER", "ollama"),
                model=os.getenv("LCCC_LLM_MODEL", "qwen3:14b"),
                models=[m.strip() for m in os.getenv("LCCC_LLM_MODELS", "").split(",") if m.strip()],
                timeout_s=int(os.getenv("LCCC_LLM_TIMEOUT", "120")),
                max_context_chars=int(os.getenv("LCCC_LLM_MAX_CONTEXT", "6000")),
                consensus_min=int(os.getenv("LCCC_LLM_CONSENSUS", "0")),
                host=os.getenv("LCCC_LLM_HOST", os.getenv("OLLAMA_HOST", "")),
                start_server=os.getenv("LCCC_LLM_START_SERVER", "1").strip().lower() in ("1", "true", "yes"),
                startup_timeout_s=int(os.getenv("LCCC_LLM_STARTUP_TIMEOUT", "15")),
                use_http=os.getenv("LCCC_LLM_USE_HTTP", "1").strip().lower() in ("1", "true", "yes"),
                timeout_chars_per_second=int(os.getenv("LCCC_LLM_TIMEOUT_CPS", "20")),
                max_timeout_s=int(os.getenv("LCCC_LLM_MAX_TIMEOUT", "300")),
                retry_attempts=int(os.getenv("LCCC_LLM_RETRY", "2")),
                retry_delay_s=float(os.getenv("LCCC_LLM_RETRY_DELAY", "2.0")),
            )
        self.config = config
        self.models = config.models or [config.model]
        if not self.models:
            self.models = [config.model]
        if self.config.consensus_min <= 0:
            self.config.consensus_min = 2 if len(self.models) > 1 else 1
        self.host = self._normalize_host(self.config.host)
        self.available = self._is_available()

    # ------------------------------------------------------------------
    # Timeout helpers
    # ------------------------------------------------------------------

    def _adaptive_timeout(self, prompt: str) -> int:
        """Scale timeout with prompt length so long evidence prompts don't time out."""
        base = self.config.timeout_s
        if self.config.timeout_chars_per_second > 0:
            extra = len(prompt) // self.config.timeout_chars_per_second
        else:
            extra = 0
        return min(base + extra, self.config.max_timeout_s)

    # ------------------------------------------------------------------
    # Availability / server management (unchanged logic)
    # ------------------------------------------------------------------

    def _is_available(self) -> bool:
        if self.config.provider.lower() == "ollama":
            if self.config.use_http and self._ping_ollama():
                return True
            return shutil.which("ollama") is not None
        return False

    def _normalize_host(self, host: str) -> str:
        host = (host or "").strip()
        if not host:
            return "http://127.0.0.1:11434"
        if not host.startswith("http://") and not host.startswith("https://"):
            host = "http://" + host
        return host.rstrip("/")

    def _ping_ollama(self) -> bool:
        if not self.host:
            return False
        url = f"{self.host}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _ensure_ollama(self) -> bool:
        if self._ping_ollama():
            return True
        if not self.config.start_server:
            return False
        if shutil.which("ollama") is None:
            return False
        try:
            creationflags = 0
            if os.name == "nt":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        except Exception as e:
            logger.warning(f"Failed to start ollama server: {e}")
            return False

        deadline = time.time() + max(3, int(self.config.startup_timeout_s))
        while time.time() < deadline:
            if self._ping_ollama():
                return True
            time.sleep(0.5)
        return False

    # ------------------------------------------------------------------
    # Raw model calls — adaptive timeout + retry
    # ------------------------------------------------------------------

    def _run_ollama_http(self, model: str, prompt: str) -> Optional[str]:
        if not self._ensure_ollama():
            logger.warning("Ollama server not running. Start it with `ollama serve`.")
            return None

        url = f"{self.host}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        timeout = self._adaptive_timeout(prompt)

        for attempt in range(max(1, self.config.retry_attempts)):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                    return (data.get("response") or "").strip()
            except urllib.error.HTTPError as e:
                logger.warning(f"Ollama HTTP error: {e}")
                return None  # HTTP errors won't improve on retry
            except Exception as e:
                is_last = attempt >= self.config.retry_attempts - 1
                if is_last:
                    logger.warning(f"Ollama HTTP call failed after {attempt + 1} attempt(s): {e}")
                    return None
                logger.warning(
                    f"Ollama HTTP call failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e} "
                    f"— retrying in {self.config.retry_delay_s}s (timeout was {timeout}s)"
                )
                time.sleep(self.config.retry_delay_s)

        return None

    def _run_ollama_cli(self, model: str, prompt: str) -> Optional[str]:
        timeout = self._adaptive_timeout(prompt)
        cmd = ["ollama", "run", model]
        for attempt in range(max(1, self.config.retry_attempts)):
            try:
                proc = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout,
                )
                if proc.returncode != 0:
                    logger.warning(f"LLM call returned error: {proc.stderr.strip()}")
                    return None
                return proc.stdout.strip()
            except Exception as e:
                is_last = attempt >= self.config.retry_attempts - 1
                if is_last:
                    logger.warning(f"LLM CLI call failed after {attempt + 1} attempt(s): {e}")
                    return None
                logger.warning(
                    f"LLM CLI call failed (attempt {attempt + 1}/{self.config.retry_attempts}): {e} "
                    f"— retrying in {self.config.retry_delay_s}s"
                )
                time.sleep(self.config.retry_delay_s)
        return None

    def _run_single(self, model: str, prompt: str) -> Optional[str]:
        if not prompt:
            return None
        if self.config.provider.lower() == "ollama":
            if self.config.use_http:
                return self._run_ollama_http(model, prompt)
            return self._run_ollama_cli(model, prompt)
        return None

    # ------------------------------------------------------------------
    # JSON extraction — scratchpad-aware
    # ------------------------------------------------------------------

    def extract_json(self, text: str) -> Optional[Any]:
        if not text:
            return None

        # Strip <scratchpad>...</scratchpad> block produced by grounded prompts.
        # Must happen before the JSON search so the scratchpad text doesn't
        # confuse the greedy {.*} / [.*] regexes.
        text = re.sub(r"<scratchpad>.*?</scratchpad>", "", text, flags=re.I | re.S)

        # Strip markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)

        # Try the largest JSON object first, then largest array.
        # Using the last/longest match handles cases where the model emits
        # multiple JSON snippets (e.g. the schema echo + the actual output).
        candidates = []
        for m in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", cleaned, re.S):
            candidates.append(m.group(0))
        # Also try a simple greedy span in case of deeply nested JSON
        m_obj = re.search(r"\{.*\}", cleaned, re.S)
        if m_obj:
            candidates.append(m_obj.group(0))
        m_list = re.search(r"\[.*\]", cleaned, re.S)
        if m_list:
            candidates.append(m_list.group(0))

        # Prefer the longest valid candidate (most complete JSON wins)
        candidates.sort(key=len, reverse=True)
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # Public interface (unchanged)
    # ------------------------------------------------------------------

    def run_json(self, prompt: str) -> List[Tuple[str, Any]]:
        results: List[Tuple[str, Any]] = []
        if not prompt:
            return results
        for model in self.models:
            output = self._run_single(model, prompt)
            if not output:
                continue
            parsed = self.extract_json(output)
            if parsed is None:
                continue
            results.append((model, parsed))
        return results