import re
from collections import defaultdict
import asyncio
import aiohttp
import numpy as np

polymer_abbrev_dict = {
     "PI": {"name": "polyisoprene", "contexts": ["microstructure", "rubber", "cis", "trans"]},
    "1,4-PI": {"name": "polyisoprene", "contexts": ["microstructure", "rubber", "cis", "trans"]},
    "3,4-PI": {"name": "polyisoprene", "contexts": ["microstructure", "rubber"]},
    "cis-PI": {"name": "polyisoprene", "contexts": ["rubber", "cis"]},
    "trans-PI": {"name": "polyisoprene", "contexts": ["rubber", "trans"]},
    "PMMA": {"name": "polymethyl methacrylate", "contexts": ["film", "resin"]},
    "PEMA": {"name": "polyethyl methacrylate", "contexts": []},
    "PS": {"name": "polystyrene", "contexts": []},
    "PB": {"name": "polybutadiene", "contexts": ["rubber", "copolymer"]},
    "ABS": {"name": "acrylonitrile butadiene styrene", "contexts": ["copolymer", "thermoplastic"]},
    "ABS/PC": {"name": "ABS/polycarbonate blend", "contexts": ["blend"]},
    "ABS/PVC": {"name": "ABS/polyvinyl chloride blend", "contexts": ["blend"]},
    "PA": {"name": "polyamide (nylon)", "contexts": ["engineering plastic"]},
    "PA6": {"name": "polyamide 6", "contexts": ["engineering plastic"]},
    "PA66": {"name": "polyamide 6,6", "contexts": ["engineering plastic"]},
    "PA11": {"name": "polyamide 11", "contexts": ["engineering plastic"]},
    "PA12": {"name": "polyamide 12", "contexts": ["engineering plastic"]},
    "PBT": {"name": "polybutylene terephthalate", "contexts": ["thermoplastic"]},
    "PC": {"name": "polycarbonate", "contexts": ["engineering plastic"]},
    "POM": {"name": "polyoxymethylene (acetal)", "contexts": ["engineering plastic"]},
    "PP": {"name": "polypropylene", "contexts": ["thermoplastic"]},
    "HDPE": {"name": "high density polyethylene", "contexts": ["thermoplastic"]},
    "LDPE": {"name": "low density polyethylene", "contexts": ["thermoplastic"]},
    "LLDPE": {"name": "linear low density polyethylene", "contexts": ["thermoplastic"]},
    "MDPE": {"name": "medium density polyethylene", "contexts": ["thermoplastic"]},
    "UHMWPE": {"name": "ultra high molecular weight polyethylene", "contexts": ["thermoplastic"]},
    "PET": {"name": "polyethylene terephthalate", "contexts": ["thermoplastic", "fiber"]},
    "PLA": {"name": "polylactic acid", "contexts": ["biodegradable", "biopolymer"]},
    "PCL": {"name": "polycaprolactone", "contexts": ["biodegradable", "biopolymer"]},
    "PHBV": {"name": "poly(3-hydroxybutyrate-co-3-hydroxyvalerate)", "contexts": ["biopolymer"]},
    "PTFE": {"name": "polytetrafluoroethylene", "contexts": ["fluoropolymer", "Teflon"]},
    "FEP": {"name": "fluorinated ethylene propylene", "contexts": ["fluoropolymer"]},
    "PFA": {"name": "perfluoroalkoxy polymer", "contexts": ["fluoropolymer"]},
    "PVDF": {"name": "polyvinylidene fluoride", "contexts": ["fluoropolymer"]},
    "PVF": {"name": "polyvinyl fluoride", "contexts": ["fluoropolymer"]},
    "PCTFE": {"name": "polychlorotrifluoroethylene", "contexts": ["fluoropolymer"]},
    "ETFE": {"name": "ethylene tetrafluoroethylene", "contexts": ["fluoropolymer"]},
    "PEEK": {"name": "polyether ether ketone", "contexts": ["high performance", "engineering plastic"]},
    "PEI": {"name": "polyetherimide", "contexts": ["engineering plastic"]},
    "PES": {"name": "polyethersulfone", "contexts": ["engineering plastic"]},
    "PSU": {"name": "polysulfone", "contexts": ["engineering plastic"]},
    "PPO": {"name": "polyphenylene oxide", "contexts": ["engineering plastic"]},
    "PPS": {"name": "polyphenylene sulfide", "contexts": ["engineering plastic"]},
    "PU": {"name": "polyurethane", "contexts": ["elastomer"]},
    "TPU": {"name": "thermoplastic polyurethane", "contexts": ["elastomer"]},
    "TPE": {"name": "thermoplastic elastomer", "contexts": ["elastomer"]},
    "TPE-O": {"name": "thermoplastic elastomer – olefinic", "contexts": ["elastomer"]},
    "TPE-S": {"name": "thermoplastic elastomer – styrenic", "contexts": ["elastomer"]},
    "TPV": {"name": "thermoplastic vulcanizate", "contexts": ["elastomer"]},
    "SBS": {"name": "styrene butadiene styrene", "contexts": ["TPE"]},
    "SEBS": {"name": "styrene ethylene butylene styrene", "contexts": ["TPE"]},
    "SIS": {"name": "styrene isoprene styrene", "contexts": ["TPE"]},
    "EVA": {"name": "ethylene vinyl acetate", "contexts": ["copolymer", "thermoplastic"]},
    "EPDM": {"name": "ethylene propylene diene monomer rubber", "contexts": ["rubber", "elastomer"]},
    "EPM": {"name": "ethylene propylene rubber", "contexts": ["rubber"]},
    "EPR": {"name": "ethylene propylene rubber", "contexts": ["rubber"]},
    "CA": {"name": "cellulose acetate", "contexts": ["biopolymer"]},
    "CAB": {"name": "cellulose acetate butyrate", "contexts": ["biopolymer"]},
    "CAP": {"name": "cellulose acetate propionate", "contexts": ["biopolymer"]},
    "CMC": {"name": "carboxymethyl cellulose", "contexts": ["biopolymer"]},
    "CTA": {"name": "cellulose triacetate", "contexts": ["biopolymer"]},
    "CN": {"name": "cellulose nitrate", "contexts": ["biopolymer"]},
    "SAN": {"name": "styrene acrylonitrile copolymer", "contexts": ["copolymer"]},
    "SB": {"name": "styrene butadiene copolymer", "contexts": ["rubber"]},
    "AES": {"name": "acrylonitrile ethylene propylene diene styrene", "contexts": ["copolymer", "rubber"]},
    "ASA": {"name": "acrylonitrile styrene acrylate", "contexts": ["copolymer", "thermoplastic"]},
    "EAA": {"name": "ethylene acrylic acid copolymer", "contexts": ["copolymer"]},
    "EMAC": {"name": "ethylene methyl acrylate copolymer", "contexts": ["copolymer"]},
    "ENBA": {"name": "ethylene n-butyl acrylate copolymer", "contexts": ["copolymer"]},
    "E/VA": {"name": "ethylene vinyl acetate copolymer", "contexts": ["copolymer"]},
    "E/P": {"name": "ethylene propylene copolymer", "contexts": ["elastomer"]},
}

POLY_PAREN_RE = re.compile(r"\bpoly\s*\(([^)]+)\)", re.IGNORECASE)
POLY_PREFIX_RE = re.compile(r"\bpoly[a-zA-Z\-]+\b", re.IGNORECASE)
MICROSTRUCTURE_RE = re.compile(r"(\d,\d-[A-Z]{1,2}|cis-PI|trans-PI)", re.IGNORECASE)  
ABBREV_RE = re.compile(r"\b([A-Z]{1,6}(?:-\d{1,2})?)\b")
EXCLUSION_RE = re.compile(r"\b(MPa|kPa|Pa|mol|wt%|polynomial|polydispersity|polymer[s]?)\b", re.IGNORECASE)

def normalize_name(name: str) -> str:
    name = name.lower().strip()
    return re.sub(r'\s+', ' ', name)

def strip_poly_prefix(name: str) -> str:
    name = normalize_name(name)
    if name.startswith("poly"):
        return name[4:]
    return name

def resolve_abbreviation(token: str, context_window: str) -> str | None:
    entry = polymer_abbrev_dict.get(token)
    if not entry:
        return None
    if entry["contexts"]:
        if any(ctx.lower() in context_window.lower() for ctx in entry["contexts"]):
            return entry["name"]
        return None
    return entry["name"]

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
        tasks = [check_pubchem(session, normalize_name(n.split("(")[0].strip())) for n in polymer_names]
        return await asyncio.gather(*tasks)

def extract_all_polymers(text: str) -> tuple[dict, dict]:
    counts = defaultdict(int)
    microstructures = defaultdict(list)
    text_clean = EXCLUSION_RE.sub("", text)

    for match in POLY_PAREN_RE.findall(text_clean):
        name = normalize_name(f"poly{match}")
        counts[name] += len(re.findall(re.escape(match), text_clean, re.IGNORECASE))

    for match in POLY_PREFIX_RE.findall(text_clean):
        if EXCLUSION_RE.match(match):
            continue
        counts[normalize_name(match)] += len(re.findall(re.escape(match), text_clean, re.IGNORECASE))

    for micro in MICROSTRUCTURE_RE.findall(text_clean):
        matched_base = None
        for abbrev, entry in polymer_abbrev_dict.items():
            if abbrev.upper().endswith(micro.split("-")[-1].upper()):
                matched_base = entry["name"]
                break
        if matched_base:
            micro_name = f"{matched_base} ({micro})"
            counts[micro_name] += 1  
            microstructures[matched_base].append(micro)

    for token_match in ABBREV_RE.finditer(text_clean):
        token = token_match.group(0)
        start = max(0, token_match.start() - 30)
        end = min(len(text_clean), token_match.end() + 30)
        context_window = text_clean[start:end]
        resolved = resolve_abbreviation(token, context_window)
        if resolved:
            counts[normalize_name(resolved)] += len(re.findall(re.escape(token), text_clean, re.IGNORECASE))

    return dict(sorted(counts.items(), key=lambda x: -x[1])), microstructures

def validate_polymers(polymer_dict: dict, micro_dict: dict) -> dict:
    polymer_names = list(polymer_dict.keys())
    results = asyncio.run(validate_polymers_async(polymer_names))
    validated = {name: count for name, valid in zip(polymer_names, results) if valid for name, count in polymer_dict.items() if name == name}
    return dict(sorted(validated.items(), key=lambda x: -x[1]))

def select_main_polymers(validated_polymer_dict: dict, micro_dict: dict) -> dict:
    if not validated_polymer_dict:
        return {}

    base_counts = defaultdict(int)
    for name, count in validated_polymer_dict.items():
        base_name = name.split("(")[0].strip()
        base_counts[base_name] += count

    sorted_bases = sorted(base_counts.items(), key=lambda x: -x[1])
    base_names = [b for b, c in sorted_bases]
    counts = np.array([c for b, c in sorted_bases], dtype=float)

    if len(counts) == 1:
        main_bases = base_names
    else:
        ratios = counts[:-1] / counts[1:]
        gap_index = np.argmax(ratios)
        main_bases = base_names[:gap_index + 1]

    main_polymers = {}
    for base in main_bases:
        if base in micro_dict:
            for micro in micro_dict[base]:
                micro_name = f"{base} ({micro})"
                if micro_name in validated_polymer_dict:
                    main_polymers[micro_name] = validated_polymer_dict[micro_name]
                else:
                    main_polymers[micro_name] = validated_polymer_dict.get(base, 0)
        else:
            if base in validated_polymer_dict:
                main_polymers[base] = validated_polymer_dict[base]

    return dict(sorted(main_polymers.items(), key=lambda x: -x[1]))

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
