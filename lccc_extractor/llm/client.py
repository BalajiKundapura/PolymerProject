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
    model: str = "qwen2.5:7b"
    models: List[str] = field(default_factory=list)
    timeout_s: int = 60
    max_context_chars: int = 1600
    consensus_min: int = 0
    host: str = ""
    start_server: bool = True
    startup_timeout_s: int = 15
    use_http: bool = True


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        if config is None:
            config = LLMConfig(
                provider=os.getenv("LCCC_LLM_PROVIDER", "ollama"),
                model=os.getenv("LCCC_LLM_MODEL", "qwen2.5:7b"),
                models=[m.strip() for m in os.getenv("LCCC_LLM_MODELS", "").split(",") if m.strip()],
                timeout_s=int(os.getenv("LCCC_LLM_TIMEOUT", "60")),
                max_context_chars=int(os.getenv("LCCC_LLM_MAX_CONTEXT", "1600")),
                consensus_min=int(os.getenv("LCCC_LLM_CONSENSUS", "0")),
                host=os.getenv("LCCC_LLM_HOST", os.getenv("OLLAMA_HOST", "")),
                start_server=os.getenv("LCCC_LLM_START_SERVER", "1").strip().lower() in ("1", "true", "yes"),
                startup_timeout_s=int(os.getenv("LCCC_LLM_STARTUP_TIMEOUT", "15")),
                use_http=os.getenv("LCCC_LLM_USE_HTTP", "1").strip().lower() in ("1", "true", "yes"),
            )
        self.config = config
        self.models = config.models or [config.model]
        if not self.models:
            self.models = [config.model]
        if self.config.consensus_min <= 0:
            self.config.consensus_min = 2 if len(self.models) > 1 else 1
        self.host = self._normalize_host(self.config.host)
        self.available = self._is_available()

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

    def _run_ollama_http(self, model: str, prompt: str) -> Optional[str]:
        if not self._ensure_ollama():
            logger.warning("Ollama server not running. Start it with `ollama serve`.")
            return None
        url = f"{self.host}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                return (data.get("response") or "").strip()
        except urllib.error.HTTPError as e:
            logger.warning(f"Ollama HTTP error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Ollama HTTP call failed: {e}")
            return None

    def _run_ollama_cli(self, model: str, prompt: str) -> Optional[str]:
        cmd = ["ollama", "run", model]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.timeout_s,
            )
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return None
        if proc.returncode != 0:
            logger.warning(f"LLM call returned error: {proc.stderr.strip()}")
            return None
        return proc.stdout.strip()

    def _run_single(self, model: str, prompt: str) -> Optional[str]:
        if not prompt:
            return None
        if self.config.provider.lower() == "ollama":
            if self.config.use_http:
                return self._run_ollama_http(model, prompt)
            return self._run_ollama_cli(model, prompt)
        return None

    def extract_json(self, text: str) -> Optional[Any]:
        if not text:
            return None
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
        candidates = []
        m_obj = re.search(r"\{.*\}", cleaned, re.S)
        if m_obj:
            candidates.append(m_obj.group(0))
        m_list = re.search(r"\[.*\]", cleaned, re.S)
        if m_list:
            candidates.append(m_list.group(0))
        for cand in candidates:
            try:
                return json.loads(cand)
            except Exception:
                continue
        return None

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
