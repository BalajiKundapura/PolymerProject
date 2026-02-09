import re
from collections import defaultdict
import asyncio

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None


nlp = None
if spacy is not None:
    try:
        nlp = spacy.load("en_ner_craft_md")
    except Exception:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None


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
    "PEO": {"name": "poly(ethylene oxide)", "contexts": []},
    "PPO": {"name": "poly(propylene oxide)", "contexts": []},

}


POLY_PAREN_RE = re.compile(r"\bpoly\s*\(([^)]+)\)", re.IGNORECASE)
POLY_PREFIX_RE = re.compile(r"\bpoly[a-zA-Z\-]+\b", re.IGNORECASE)
ABBREV_RE = re.compile(r"\b([A-Z]{2,6}(?:-\d{1,2})?)\b")
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


def extract_polymers(text: str) -> dict:
    counts = defaultdict(int)
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

    for token_match in ABBREV_RE.finditer(text_clean):
        token = token_match.group(0)
        start = max(0, token_match.start() - 50)
        end = min(len(text_clean), token_match.end() + 50)
        context_window = text_clean[start:end]
        resolved = resolve_abbreviation(token, context_window)
        if resolved:
            name = normalize_name(resolved)
            counts[name] += 1

    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("CHEMICAL", "POLYMER"):
                counts[normalize_name(ent.text)] += 1

    counts = {k: v for k, v in counts.items() if not EXCLUSION_RE.match(k)}
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


VALIDATION_CACHE = {}
KNOWN_POLYMERS = {normalize_name(p["name"]) for p in polymer_abbrev_dict.values()}
PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON"
CONCURRENCY = 30
TIMEOUT = aiohttp.ClientTimeout(total=4) if aiohttp is not None else None

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
    if aiohttp is None:
        return [True for _ in polymer_names]

    connector = aiohttp.TCPConnector(limit=CONCURRENCY, ssl=False)
    async with aiohttp.ClientSession(timeout=TIMEOUT, connector=connector) as session:
        tasks = [check_pubchem(session, normalize_name(n)) for n in polymer_names]
        return await asyncio.gather(*tasks)

def validate_polymers(polymer_dict: dict) -> dict:
    if aiohttp is None:
        # Validation is optional; keep extraction usable without network deps installed.
        return polymer_dict

    polymer_names = list(polymer_dict.keys())
    results = asyncio.run(validate_polymers_async(polymer_names))
    validated = {name: polymer_dict[name] for name, valid in zip(polymer_names, results) if valid}
    return dict(sorted(validated.items(), key=lambda x: -x[1]))


def select_main_polymers(validated_polymer_dict: dict, threshold_ratio: float = 0.5) -> dict:

    if not validated_polymer_dict:
        return {}
    sorted_polymers = sorted(validated_polymer_dict.items(), key=lambda x: -x[1])
    max_count = sorted_polymers[0][1]
    main_polymers = {name: count for name, count in sorted_polymers if count >= max_count * threshold_ratio}
    return main_polymers


if __name__ == "__main__":
    file_path = "rawData/paper.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        sample_text = f.read()

    all_polymers = extract_polymers(sample_text) 
    print("All extracted polymers (before validation):")
    for polymer, count in all_polymers.items():
        print(f"{polymer}: {count}")

    validated_polymers = validate_polymers(all_polymers)
    print("\nValidated polymers (PubChem):")
    for polymer, count in validated_polymers.items():
        print(f"{polymer}: {count}")

    main_polymers = select_main_polymers(validated_polymers, threshold_ratio=0.5)
    print("\nMain polymer topics (frequency-distinguished):")
    for polymer, count in main_polymers.items():
        print(f"{polymer}: {count}")
