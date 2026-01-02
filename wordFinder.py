import re
from collections import defaultdict

polymerAbreDict = {
    "ABA": "Acrylonitrile Butadiene Acrylate",
    "ABAK": "Acrylonitrile Butadiene Acrylate",
    "ABS": "Acrylonitrile Butadiene Styrene",
    "ABS + PUR": "Blend of Acrylonitrile Butadiene Styrene and Polyurethane",
    "ABS / PVC": "Acrylonitrile Butadiene Styrene / Polyvinyl Chloride",
    "ACS": "Acrylonitrile Chlorinated Polyethylene Styrene",
    "AES": "Acrylonitrile Ethylene-Propylenediene Styrene",
    "AMA": "Acrylate Maleic Anhydride Terpolymer",
    "AMMA": "Acrylonitrile Methyl Methacrylate",
    "APO": "Amorphous Polyolefin",
    "AS": "Acrylonitrile Styrene",
    "ASA": "Acrylonitrile Styrene Acrylate",
    "ASA + PC": "Blend of Acrylonitrile Styrene Acrylate and Polycarbonate",
    "BMC": "Bulk Molding Compound",
    "BMI": "Bis Maleimide",
    "BR": "Butadiene Rubber",
    "CA": "Cellulose Acetate",
    "CAB": "Cellulose Acetate Butyrate",
    "CAP": "Cellulose Acetate Propionate",
    "CF": "Cresole-Formaldehyde",
    "CMC": "Carboxylmethyl Cellulose",
    "CN": "Cellulose Nitrate (Celluloid)",
    "COC": "Cycloolefin Copolymer",
    "COP": "Copolyester Thermoplastic Elastomer",
    "CP": "Cellulose Propionate",
    "CPE": "Chlorinated Polyethylene",
    "CPVC": "Chlorinated Polyvinyl Chloride",
    "CSF": "Casein-Formaldehyde",
    "CTA": "Cellulose Triacetate",
    "CTFE": "Chlorotrifluoroethylene",
    "DAP": "Diallyl Phthalate (Thermoset)",
    "EAA": "Ethylene Acrylic Acid Copolymer",
    "EBA": "Ethylene Butylacetate",
    "ECTFE": "Ethylene Chlorotrifluoroethylene",
    "EEAK": "Ethylene Ethylacrylate",
    "EMA": "Ethylene Methacrylic Acid",
    "EMAA": "Ethylene Methacrylic Acid Copolymer",
    "EMAC": "Ethylene Methyl Acrylate Copolymer",
    "EP": "Ethylene Propylene",
    "EPDM": "Ethylene Propylene Diene Monomer Rubber",
    "EPM": "Ethylene Propylene Copolymer Rubber",
    "EPR": "Ethylene Propylene Rubber",
    "EPS": "Expandable Polystyrene",
    "ESBR": "Emulsion Styrene Butadiene Rubber",
    "EVA": "Ethylene Vinyl Acetate",
    "EVOH": "Ethylene Vinyl Alcohol",
    "FEP": "Fluorinated Ethylene Propylene",
    "FF": "Furan-Formaldehyde",
    "FRP": "Fibre Reinforced Plastic",
    "GPPS": "General Purpose Polystyrene",
    "HDPE": "High Density Polyethylene",
    "HIPS": "High Impact Polystyrene",
    "HMC": "High Strength Molding Compound",
    "HMWHDPE": "High Molecular Weight High Density Polyethylene",
    "ION": "Ionomer",
    "IPN": "Interpenetrating Polymer Network",
    "IR": "Polyisoprene",
    "LCP": "Liquid Crystal Polymer",
    "LDPE": "Low Density Polyethylene",
    "LPE": "Linear Polyethylene",
    "MA": "Maleic Anhydride",
    "MABS": "Methylmethacrylate Acrylonitrile Butadiene Styrene",
    "MBS": "Methacrylate Butadiene Styrene",
    "MC": "Methyl Cellulose",
    "MDPE": "Medium Density Polyethylene",
    "MF": "Melamine Formaldehyde",
    "MP": "Melamine Phenolic",
    "MPF": "Melamine Phenol-Formaldehyde",
    "NBR": "Nitrile Butadiene Rubber",
    "OSA": "Olefin Modified Styrene Acrylonitrile",
    "P": "Phenolic",
    "PA": "Polyamide",
    "PA 11": "Polyamide 11",
    "PA 12": "Polyamide 12",
    "PA 46": "Polyamide 46",
    "PA 6": "Polyamide 6",
    "PA 66": "Polyamide 66",
    "PBT": "Polybutylene Terephthalate",
    "PC": "Polycarbonate",
    "PE": "Polyethylene",
    "PEEK": "Polyetheretherketone",
    "PEI": "Polyetherimide",
    "PEN": "Polyethylene Naphthalate",
    "PET": "Polyethylene Terephthalate",
    "PF": "Phenol-Formaldehyde",
    "PI": "Polyimide",
    "PLA": "Poly Lactic Acid",
    "PMMA": "Polymethyl Methacrylate",
    "POM": "Polyoxymethylene (Acetal)",
    "PP": "Polypropylene",
    "PPE": "Polyphenylene Ether",
    "PPS": "Polyphenylene Sulfide",
    "PS": "Polystyrene",
    "PTFE": "Polytetrafluoroethylene",
    "PU": "Polyurethane",
    "PVC": "Polyvinyl Chloride",
    "SAN": "Styrene Acrylonitrile",
    "SBR": "Styrene Butadiene Rubber",
    "SEBS": "Styrene Ethylene Butylene Styrene",
    "SIS": "Styrene Isoprene Styrene",
    "SMA": "Styrene Maleic Anhydride",
    "SMC": "Sheet Molding Compound",
    "TPE": "Thermoplastic Elastomer",
    "TPU": "Thermoplastic Urethane",
    "UHMWPE": "Ultra High Molecular Weight Polyethylene",
    "VLDPE": "Very Low Density Polyethylene",
    "XLPE": "Crosslinked Polyethylene"
}

POLY_PAREN_RE = re.compile(r"\bpoly\s*\([^)]+\)", re.IGNORECASE)
POLY_PREFIX_RE = re.compile(r"\bpoly[a-zA-Z][a-zA-Z\-]+\b", re.IGNORECASE)
ABBREV_RE = re.compile(r"\b[A-Z]{1,6}(?:\s\d{1,2})?s?\b")
BLEND_RE = re.compile(r"\b[A-Z0-9 ]+(?:\/|\+)[A-Z0-9 ]+\b")
PLURAL_POLY_RE = re.compile(r"\b(poly[a-zA-Z\-]+)s\b", re.IGNORECASE)
PLURAL_ABBREV_RE = re.compile(r"\b([A-Z]{1,6}(?:\s\d{1,2})?)s\b")
EXCLUSION_RE = re.compile(r"\b(MPa|kPa|Pa|mol|wt%)\b")

def normalize_plural(token: str) -> str:
    token = token.strip()
    if PLURAL_POLY_RE.match(token) or PLURAL_ABBREV_RE.match(token):
        token = token[:-1]
    return token

def normalize_polymer_name(name: str) -> str:
    """Normalize polymer names into a standard plain format."""
    n = name.strip()
    n = n.lower()  # lowercase
    n = re.sub(r"poly\((.*?)\)", r"poly\1", n)  # remove parentheses
    n = re.sub(r"(?<!\bpi)(s)$", "", n)  # remove plural 's' (but not 1,4-PI)
    n = " ".join(n.split())  # remove extra spaces
    return n

def resolve_polymer(token: str) -> str | None:
    token = token.strip()
    if re.match(r'(?i)^poly', token):
        return normalize_polymer_name(token)
    if '/' in token or '+' in token:
        return None
    resolved = polymerAbreDict.get(token)
    if resolved:
        return normalize_polymer_name(resolved)
    return None

def extract_polymers(text: str) -> dict:
    counts = defaultdict(int)
    text = EXCLUSION_RE.sub("", text)
    candidates = set()
    candidates.update(POLY_PAREN_RE.findall(text))
    candidates.update(POLY_PREFIX_RE.findall(text))
    candidates.update(ABBREV_RE.findall(text))
    candidates.update(BLEND_RE.findall(text))

    for raw in candidates:
        raw = normalize_plural(raw)
        parts = [raw]
        if '/' in raw or '+' in raw:
            parts = re.split(r"[\/\+]", raw)
        for part in parts:
            part = part.strip()
            resolved = resolve_polymer(part)
            if resolved:
                counts[resolved] += text.lower().count(part.lower())
    return dict(counts)

if __name__ == "__main__":
    file_path = r"C:\Users\Balaji-Personal\Desktop\PolymerProject-1\rawData\paper.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        sample_text = f.read()

    result = extract_polymers(sample_text)
    print("Detected polymers:")
    for polymer, count in result.items():
        print(f"{polymer}: {count}")
