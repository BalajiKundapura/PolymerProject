import re
import sys
import os
from typing import List, Dict, Any

# Ensure parent package is importable when running this module directly.
if __package__ is None:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

from polymerSubject import extract_all_polymers

# ============================================================
# HARD BLOCKERS (units that mean NOT solvent ratios)
# ============================================================

UNIT_BLOCK_RE = re.compile(
    r'kg\s*/\s*mol|kg\s*mol|molar\s+mass|M_n|M_w|M_p|SEC|NMR',
    re.IGNORECASE
)

TABLE_CAPTION_RE = re.compile(
    r'\b(Table|Figure|Scheme)\s*\d+|dx\.doi|Macromolecules',
    re.IGNORECASE
)

# ============================================================
# SOLVENT & RATIO PATTERNS
# ============================================================

SOLVENT_SYSTEM_RE = re.compile(
    r'([A-Za-z][A-Za-z\-]{2,})\s*/\s*([A-Za-z][A-Za-z\-]{2,})'
)

RATIO_RE = re.compile(
    r'(\d{1,3})\s*[:/]\s*(\d{1,3})'
)

SOLVENT_CUES = re.compile(
    r'mobile\s+phase|eluent|solvent|composition|mixture',
    re.IGNORECASE
)

# ============================================================
# CONTEXT TRACKER
# ============================================================

class ConditionContext:
    def __init__(self):
        self.active_polymer = None
        self.active_solvent = None
        self.active_ratio = None

# ============================================================
# CORE EXTRACTION
# ============================================================

def extract_conditions(
    sentences: List[str],
    lookback: int = 6
) -> List[Dict[str, Any]]:

    context = ConditionContext()
    events = []

    for idx, sentence in enumerate(sentences):

        sentence_l = sentence.lower()

        # -----------------------------------
        # SOLVENT SYSTEM
        # -----------------------------------
        solvent_system = None
        for a, b in SOLVENT_SYSTEM_RE.findall(sentence):
            if not UNIT_BLOCK_RE.search(a) and not UNIT_BLOCK_RE.search(b):
                solvent_system = f"{a}/{b}"
                context.active_solvent = solvent_system
                break

        # -----------------------------------
        # RATIO (CONTEXT-AWARE, SAFE)
        # -----------------------------------
        ratio = None
        if not UNIT_BLOCK_RE.search(sentence):
            m = RATIO_RE.search(sentence)

            # ✅ FIXED LOGIC + CONTEXTUAL ADAPTIVITY
            if m and (
                solvent_system
                or SOLVENT_CUES.search(sentence)
                or context.active_solvent
            ):
                a, b = int(m.group(1)), int(m.group(2))
                if 1 <= a <= 100 and 1 <= b <= 100:
                    ratio = f"{a}/{b}"
                    context.active_ratio = ratio

        # -----------------------------------
        # POLYMER ASSIGNMENT
        # -----------------------------------
        polymer = None
        confidence = 0.0
        context_sentences = []

        found, _ = extract_all_polymers(sentence)
        if found:
            polymer = max(found, key=found.get)
            confidence = 0.95
            context.active_polymer = polymer
            context_sentences.append(sentence)
        else:
            for j in range(idx - 1, max(-1, idx - lookback), -1):
                prev = sentences[j]
                context_sentences.append(prev)
                found_prev, _ = extract_all_polymers(prev)
                if found_prev:
                    polymer = max(found_prev, key=found_prev.get)
                    confidence = max(0.6, 0.9 - 0.1 * (idx - j))
                    context.active_polymer = polymer
                    break

        if not polymer:
            polymer = context.active_polymer or "UNKNOWN"
            confidence = 0.5 if polymer != "UNKNOWN" else 0.3

        # -----------------------------------
        # IMPLICIT REFERENCES
        # -----------------------------------
        if "these conditions" in sentence_l:
            solvent_system = solvent_system or context.active_solvent
            ratio = ratio or context.active_ratio

        # -----------------------------------
        # EVENT CREATION
        # -----------------------------------
        if solvent_system or ratio:
            events.append({
                "sentence_index": idx,
                "polymer": polymer,
                "confidence": round(confidence, 2),
                "solvent_system": solvent_system or context.active_solvent,
                "ratio": ratio or context.active_ratio,
                "ratio_type": "solvent_composition" if ratio else None,
                "sentence": sentence.strip(),
                "context": context_sentences[::-1]
            })

    return events

# ============================================================
# TEST HARNESS
# ============================================================

if __name__ == "__main__":
    from txtNormalize import preprocess_file

    sentences = preprocess_file("rawData/paper.txt")
    results = extract_conditions(sentences)

    for r in results:
        print("\n--- CONDITION FOUND ---")
        for k, v in r.items():
            print(f"{k}: {v}")
"""Package initializer for testExtractor."""
"""Package initializer for testExtractor."""
