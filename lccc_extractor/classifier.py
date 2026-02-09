import os
import re
from typing import Optional

from .logging_config import logger
from .patterns import LCCC_POS_CUES, NON_LCCC_EXCLUDE

try:
    from sentence_transformers import SentenceTransformer

    _MODEL_NAME = os.getenv("LCCC_SENT_MODEL", "all-MiniLM-L6-v2")
except Exception:
    SentenceTransformer = None
    _MODEL_NAME = None


_CLASSIFIER_INSTANCE: Optional["ParagraphClassifier"] = None


def get_classifier() -> "ParagraphClassifier":
    global _CLASSIFIER_INSTANCE
    if _CLASSIFIER_INSTANCE is None:
        use_semantic = os.getenv("LCCC_USE_SEMANTIC", "0").strip().lower() in ("1", "true", "yes")
        _CLASSIFIER_INSTANCE = ParagraphClassifier(use_semantic=use_semantic)
    return _CLASSIFIER_INSTANCE


class ParagraphClassifier:
    def __init__(self, use_semantic: bool = False) -> None:
        self.model = None
        if use_semantic and SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(_MODEL_NAME)
                logger.info(f"Loaded sentence-transformers: {_MODEL_NAME}")
            except Exception as e:
                logger.warning(f"Semantic model load failed, fallback to heuristics: {e}")

    def is_lccc(self, text: str) -> bool:
        """Determine whether the text is describing LCCC / critical conditions."""
        t = (text or "").lower()

        def exclude_hit(term: str) -> bool:
            term_l = term.lower().strip()
            if not term_l:
                return False
            # Short technique acronyms should match as whole words (avoid "sec" in "section").
            if term_l.isalpha() and len(term_l) <= 4:
                return bool(re.search(rf"\b{re.escape(term_l)}\b", t))
            return term_l in t

        if any(exclude_hit(k) for k in NON_LCCC_EXCLUDE):
            if not any(k.lower() in t for k in LCCC_POS_CUES):
                return False

        # Common abbreviation in polymer chromatography literature
        if re.search(r"\bcap\b", t) and ("critical" in t or "adsorption" in t or "column" in t or "chromatograph" in t):
            return True

        if any(k.lower() in t for k in LCCC_POS_CUES):
            return True

        if "critical" in t and "condition" in t and ("chromatograph" in t or "column" in t or "eluent" in t):
            return True

        return False

    def classify(self, para: str) -> str:
        """Classify paragraph as methods, results, or other."""
        p = (para or "").lower()
        method_cues = [
            "experimental",
            "materials",
            "methods",
            "procedure",
            "chromatograph",
            "column",
            "eluent",
            "mobile phase",
            "stationary phase",
            "flow rate",
            "injection",
            "detector",
            "temperature",
        ]
        result_cues = [
            "results",
            "we observed",
            "we found",
            "figure",
            "table",
            "elution",
            "peak",
            "chromatogram",
            "distribution",
            "resolution",
        ]

        score_methods = sum(p.count(k) for k in method_cues)
        score_results = sum(p.count(k) for k in result_cues)

        if self.is_lccc(para):
            score_methods += 3

        if score_methods >= max(1, score_results):
            return "methods_results"
        if score_results > 0:
            return "results"
        return "other"

