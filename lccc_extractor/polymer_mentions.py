import re
from typing import Dict, List, Set, Tuple

from polymerSubject import normalize_name, polymer_abbrev_dict

from .logging_config import logger
from .models import PolymerMention


def _build_paragraph_positions(text: str, paragraphs: List[str]) -> List[Tuple[int, int, int]]:
    positions: List[Tuple[int, int, int]] = []
    cursor = 0
    for idx, para in enumerate(paragraphs):
        found = text.find(para, cursor)
        if found == -1:
            compact_para = re.sub(r"\s+", " ", para.strip())
            compact_text = re.sub(r"\s+", " ", text[cursor:])
            match = re.search(re.escape(compact_para), compact_text)
            if match:
                start = cursor + match.start()
                end = start + len(para)
                positions.append((start, end, idx))
                cursor = end
                continue
            positions.append((cursor, cursor + len(para), idx))
            cursor += len(para)
            continue
        start = found
        end = found + len(para)
        positions.append((start, end, idx))
        cursor = end
    return positions


def _build_alias_map(known_polymers: Set[str]) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {p: [p] for p in known_polymers}
    for abbr, entry in polymer_abbrev_dict.items():
        canonical = normalize_name(entry["name"])
        if canonical in aliases:
            aliases[canonical].append(abbr)
    return aliases


def _alias_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias)
    if re.fullmatch(r"[A-Z0-9\-]{2,10}", alias):
        return re.compile(rf"\b{escaped}\b", re.I)
    return re.compile(rf"(?<!\w){escaped}(?:s)?(?!\w)", re.I)


def find_polymer_mentions(text: str, paragraphs: List[str], known_polymers: Set[str]) -> List[PolymerMention]:
    """
    Find all mentions of known polymers in text with context windows.
    Returns list of PolymerMention objects with surrounding context.
    """
    mentions: List[PolymerMention] = []
    para_positions = _build_paragraph_positions(text, paragraphs)
    alias_map = _build_alias_map(known_polymers)
    seen_mentions: Set[Tuple[int, int, str]] = set()

    for canonical, aliases in alias_map.items():
        for alias in aliases:
            pattern = _alias_pattern(alias)
            for match in pattern.finditer(text):
                start_pos = match.start()
                end_pos = match.end()
                key = (start_pos, end_pos, canonical)
                if key in seen_mentions:
                    continue
                seen_mentions.add(key)

                para_idx = None
                for p_start, p_end, idx in para_positions:
                    if p_start <= start_pos < p_end:
                        para_idx = idx
                        break

                if para_idx is None:
                    continue

                context_start = max(0, start_pos - 200)
                context_end = min(len(text), end_pos + 200)

                context_before = text[context_start:start_pos].strip()
                context_after = text[end_pos:context_end].strip()
                full_context = paragraphs[para_idx]

                mention = PolymerMention(
                    polymer_name=match.group(0),
                    canonical_name=canonical,
                    position=start_pos,
                    context_before=context_before[-150:] if len(context_before) > 150 else context_before,
                    context_after=context_after[:150] if len(context_after) > 150 else context_after,
                    full_context=full_context,
                    para_idx=para_idx,
                )
                mentions.append(mention)

    logger.info(f"Found {len(mentions)} polymer mentions in text")
    return mentions

