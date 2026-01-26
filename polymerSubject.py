import re
from collections import defaultdict
import asyncio
import aiohttp
import numpy as np


polymer_abbrev_dict = {
    "PMMA": {"name": "poly(methyl methacrylate)", "contexts": []},
    "hPMMA": {"name": "poly(methyl methacrylate)", "contexts": []},
    "dPMMA": {"name": "poly(methyl methacrylate)", "contexts": []},
    "PS": {"name": "polystyrene", "contexts": []},
    "hPS": {"name": "polystyrene", "contexts": []},
    "dPS": {"name": "polystyrene", "contexts": []},
    "PI": {"name": "polyisoprene", "contexts": []},
    "PEMA": {"name": "poly(ethyl methacrylate)", "contexts": []},
    "PVC": {"name": "poly(vinyl chloride)", "contexts": []},
    "PB": {"name": "polybutadiene", "contexts": []},
    "PPG": {"name": "poly(propylene glycol)", "contexts": []},
    "PDMA": {"name": "poly(decyl methacrylate)", "contexts": []},
    "polyTOC": {"name": "poly(1,3,6-trioxocane)", "contexts": []},
    "AA-1,2ED": {"name": "poly[(adipic acid)-co-(1,2-ethanediol)]", "contexts": []},
    "AA-1,2PD": {"name": "poly[(adipic acid)-co-(1,2-propanediol)]", "contexts": []},
    "AA-1,3BD": {"name": "poly[(adipic acid)-co-(1,3-butanediol)]", "contexts": []},
    "AA-1,3PD": {"name": "poly[(adipic acid)-co-(1,3-propanediol)]", "contexts": []},
    "AA-1,4BD": {"name": "poly[(adipic acid)-co-(1,4-butanediol)]", "contexts": []},
    "AA-DEG": {"name": "poly[(adipic acid)-co-(diethylene glycol)]", "contexts": []},
    "AA-DPG": {"name": "poly[(adipic acid)-co-(dipropylene glycol)]", "contexts": []},
    "AA-NPG": {"name": "poly[(adipic acid)-co-(neopentyl glycol)]", "contexts": []},
    "PA-DEG": {"name": "poly[(phthalic acid)-co-(diethylene glycol)]", "contexts": []},
    "PA-EG": {"name": "poly[(phthalic acid)-co-(ethylene glycol)]", "contexts": []},
    "PA-TEG": {"name": "poly[(phthalic acid)-co-(triethylene glycol)]", "contexts": []},
    "PC": {"name": "polycarbonate", "contexts": []},
    "PDEGA": {"name": "poly[di(ethylene glycol) adipate]", "contexts": []},
    "PDMS": {"name": "polydimethylsiloxane", "contexts": []},
    "PEG": {"name": "poly(ethylene glycol)", "contexts": []},
    "mPEG": {"name": "poly(ethylene glycol)", "contexts": []},
    "PEG-MME": {"name": "poly(ethylene glycol)", "contexts": []},
    "PEG-DME": {"name": "poly(ethylene glycol)", "contexts": []},
    "MeO-PEG-DME": {"name": "poly(ethylene glycol)", "contexts": []},
    "PtBMA": {"name": "poly(t-butyl methacrylate)", "contexts": []},
    "PBMA": {"name": "poly(n-butyl methacrylate)", "contexts": []},
    "PnBMA": {"name": "poly(n-butyl methacrylate)", "contexts": []},
    "P2VP": {"name": "poly(2-vinylpyridine)", "contexts": []},
    "PPOA": {"name": "poly(propylene oxide) adipate", "contexts": []},
    "PA6": {"name": "polycaprolactam", "contexts": []},
    "Nylon-6": {"name": "polycaprolactam", "contexts": []},
    "PBT": {"name": "polybutylene terephthalate", "contexts": []},
    "PBTF": {"name": "polybutylene terephthalate", "contexts": []},
    "PSU": {"name": "polysulfone", "contexts": []},
    "PPha-tere": {"name": "poly(phenolphthalein terephthalate)", "contexts": []},
    "PBA": {"name": "poly(n-butyl acrylate)", "contexts": []},
    "PnBA": {"name": "poly(n-butyl acrylate)", "contexts": []},
    "AA-HD": {"name": "poly[(adipic acid)-co-(1,6-hexanediol)]", "contexts": []},
    "AA-1,6HD": {"name": "poly[(adipic acid)-co-(1,6-hexanediol)]", "contexts": []},
    "PLLA": {"name": "poly(L-lactide)", "contexts": []},
    "PLA": {"name": "poly(L-lactide)", "contexts": []},
    "PPA": {"name": "poly(propylene adipate)", "contexts": []},
    "PTHF": {"name": "polytetrahydrofuran", "contexts": []},
    "PIB": {"name": "polyisobutylene", "contexts": []},
    "PEtOx": {"name": "poly(2-ethyl-2-oxazoline)", "contexts": []},
    "PBO": {"name": "poly(butylene oxide)", "contexts": []},
    "PHO": {"name": "poly(hexylene oxide)", "contexts": []},
    "PiBoA": {"name": "poly(isobornyl acrylate)", "contexts": []},
    "PMA": {"name": "poly(methyl acrylate)", "contexts": []},
    "PCL": {"name": "polycaprolactone", "contexts": []},
    "PCL-ME": {"name": "polycaprolactone", "contexts": []},
    "PVP": {"name": "polyvinylpyrrolidone", "contexts": []},
    "PVAc": {"name": "polyvinyl acetate", "contexts": []},
    "PVA": {"name": "polyvinyl acetate", "contexts": []},
    "PDPA": {"name": "poly(diphenolic acid)", "contexts": []},
    "AA": {"name": "adipic acid", "contexts": []},
    "AS": {"name": "adipic acid", "contexts": []},
    "PAmb": {"name": "poly(ambrettolide)", "contexts": []},
    "cPAmb": {"name": "poly(ambrettolide)", "contexts": []},
    "SPS": {"name": "polystyrene sulfonate", "contexts": []},
    "PSS": {"name": "polystyrene sulfonate", "contexts": []},
    "PAA": {"name": "poly(acrylic acid)", "contexts": []},
    "PPOPA": {"name": "poly(propylene phthalate)", "contexts": []},
    "polyDODT": {"name": "poly(3,6-dioxa-1,8-octanedithiol)", "contexts": []},
    "EP": {"name": "poly(ethylene-co-propylene)", "contexts": []},
    "EP Copolomer": {"name": "poly(ethylene-co-propylene)", "contexts": []},
    "PE": {"name": "polyethylene", "contexts": []},
    "PP": {"name": "polypropylene", "contexts": []},
    "Tween 20": {"name": "polyoxyethylene sorbitan monolaurate", "contexts": []},
    "Polysorbate 20": {"name": "polyoxyethylene sorbitan monolaurate", "contexts": []},
    "PEEC": {"name": "poly(ethylene ether carbonate)", "contexts": []},
    "PPEC": {"name": "poly(propylene ether carbonate)", "contexts": []},
    "Tween 40": {"name": "polyoxyethylene sorbitan monopalmitate", "contexts": []},
    "Polysorbate 40": {"name": "polyoxyethylene sorbitan monopalmitate", "contexts": []},
}

POLY_PAREN_RE = re.compile(r"\bpoly\s*\(([^)]+)\)", re.IGNORECASE)
POLY_PREFIX_RE = re.compile(r"\bpoly[a-zA-Z\-]+\b", re.IGNORECASE)
MICROSTRUCTURE_RE = re.compile(r"(\d,\d-[A-Z]{1,2}|cis-PI|trans-PI)", re.IGNORECASE)
ABBREV_RE = re.compile(r"\b([A-Z]{1,6}(?:-\d{1,2})?)\b")
EXCLUSION_RE = re.compile(r"\b(MPa|kPa|Pa|mol|wt%|polynomial|polydispersity|polymer[s]?)\b", re.IGNORECASE)

def normalize_name(name: str) -> str:
    name = name.lower().strip()
    if name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]
    return re.sub(r'\s+', ' ', name)

def resolve_abbreviation(token: str, context_window: str) -> str | None:
    entry = polymer_abbrev_dict.get(token)
    if not entry:
        return None
    if entry["contexts"]:
        if any(ctx.lower() in context_window.lower() for ctx in entry["contexts"]):
            return entry["name"]
        return None
    return entry["name"]


def extract_all_polymers(text: str) -> tuple[dict, dict]:
    counts = defaultdict(int)
    microstructures = defaultdict(list)
    text_clean = EXCLUSION_RE.sub("", text)  

    seen_spans = set()

    for match in POLY_PAREN_RE.finditer(text_clean):
        span = match.span()
        if span in seen_spans:
            continue
        seen_spans.add(span)
        raw_name = f"poly{match.group(1)}"
        name = normalize_name(raw_name)

        for abbrev, entry in polymer_abbrev_dict.items():
            if normalize_name(entry["name"]) == name:
                name = normalize_name(entry["name"])
                break

        counts[name] += 1

    for match in POLY_PREFIX_RE.finditer(text_clean):
        span = match.span()
        if span in seen_spans:
            continue
        seen_spans.add(span)
        raw_name = normalize_name(match.group(0))

        for abbrev, entry in polymer_abbrev_dict.items():
            if normalize_name(entry["name"]) == raw_name:
                raw_name = normalize_name(entry["name"])
                break

        counts[raw_name] += 1

    for micro_match in MICROSTRUCTURE_RE.finditer(text_clean):
        micro = micro_match.group(1)
        for abbrev, entry in polymer_abbrev_dict.items():
            base_name = normalize_name(entry["name"])
            if abbrev.upper().endswith(micro.split("-")[-1].upper()):
                counts[base_name] += 1 
                microstructures[base_name].append(micro)
                break

    for token_match in ABBREV_RE.finditer(text_clean):
        token = token_match.group(0)
        start = max(0, token_match.start() - 30)
        end = min(len(text_clean), token_match.end() + 30)
        context_window = text_clean[start:end]
        resolved = resolve_abbreviation(token, context_window)
        if resolved:
            name = normalize_name(resolved)
            counts[name] += 1

    counts = {k: v for k, v in counts.items() if not EXCLUSION_RE.match(k)}

    return dict(sorted(counts.items(), key=lambda x: -x[1])), microstructures


VALIDATION_CACHE = {}
KNOWN_POLYMERS = {normalize_name(p["name"]) for p in polymer_abbrev_dict.values()}
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON"
CONCURRENCY = 30
TIMEOUT = aiohttp.ClientTimeout(total=4)

async def check_pubchem(session, name: str) -> bool:
    if name in VALIDATION_CACHE:
        return VALIDATION_CACHE[name]
    if name in KNOWN_POLYMERS:
        VALIDATION_CACHE[name] = True
        return True
    try:
        async with session.get(PUBCHEM_URL.format(name)) as r:
            if r.status != 200:
                VALIDATION_CACHE[name] = False
                return False
            data = await r.json()
            valid = "IdentifierList" in data
            VALIDATION_CACHE[name] = valid
            return valid
    except Exception:
        VALIDATION_CACHE[name] = False
        return False

async def validate_polymers_async(polymer_names):
    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(timeout=TIMEOUT, connector=connector) as session:
        tasks = [
            check_pubchem(session, re.sub(r"\(.*\)", "", normalize_name(n)).strip())
            for n in polymer_names
        ]
        return await asyncio.gather(*tasks)

def validate_polymers(polymer_dict: dict, micro_dict: dict) -> dict:
    polymer_names = list(polymer_dict.keys())
    results = asyncio.run(validate_polymers_async(polymer_names))
    validated = {name: count for name, valid in zip(polymer_names, results) if valid for name, count in polymer_dict.items() if name == name}
    return dict(sorted(validated.items(), key=lambda x: -x[1]))

def select_main_polymers(validated_polymer_dict: dict, micro_dict: dict) -> dict:

    if not validated_polymer_dict:
        return {}

    sorted_polymers = sorted(validated_polymer_dict.items(), key=lambda x: -x[1])
    names = [name for name, count in sorted_polymers]
    counts = [count for name, count in sorted_polymers]

    if len(counts) == 1:
        return {names[0]: counts[0]}

    ratios = [counts[i] / counts[i + 1] for i in range(len(counts) - 1)]
    gap_index = np.argmax(ratios)

    main_names = names[:gap_index + 1]
    main_polymers = {name: validated_polymer_dict[name] for name in main_names}

    return main_polymers


if __name__ == "__main__":
    file_path = r"C:\Users\Balaji-Personal\Desktop\PolymerProject-1\rawData\paper.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        sample_text = f.read()

    all_polymers, micro_dict = extract_all_polymers(sample_text)
    print("All extracted polymers (before validation):")
    for polymer, count in all_polymers.items():
        print(f"{polymer}: {count}")

    validated_polymers = validate_polymers(all_polymers, micro_dict)
    print("\nValidated polymers (PubChem):")
    for polymer, count in validated_polymers.items():
        print(f"{polymer}: {count}")

    main_polymers = select_main_polymers(validated_polymers, micro_dict)
    print("\nMain polymer topics (deterministic with microstructures):")
    for polymer, count in main_polymers.items():
        print(f"{polymer}: {count}")