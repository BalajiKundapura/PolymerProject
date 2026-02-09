import re


# -----------------------
# Lexicons
# -----------------------
LCCC_POS_CUES = [
    "liquid chromatography at the critical condition",
    "critical condition",
    "critical conditions",
    "lccc",
    "chromatographically invisible",
    "critical adsorption point",
    "critical point",
]

NON_LCCC_EXCLUDE = [
    "size exclusion chromatography",
    "sec",
    "gpc",
    "gel permeation chromatography",
    "ion exchange chromatography",
    "iec",
    "anion exchange",
    "cation exchange",
]

COLUMN_PHASE_TERMS = [
    "C18",
    "ODS",
    "CN",
    "cyano",
    "Diol",
    "Amino",
    "silica",
    "HILIC",
    "reversed phase",
    "normal phase",
    "grafted",
    "bonded",
]


# -----------------------
# Compiled Regex Patterns
# -----------------------
RE_TEMPERATURE = re.compile(
    r"\b(?:temp(?:erature)?|column\s*temperature|evaporator\s*temperature|T)\b\s*(?:[:=]|of|at|was|is)?\s*([+-]?\d+(?:\.\d+)?)\s*(C|F|K)\b",
    re.I,
)
RE_FLOW = re.compile(
    r"\bflow[- ]*rate\b\s*(?:[:=]|of|at|was|is|were)?\s*([0-9]+(?:\.\d+)?)\s*(mL|ml|uL)\s*/\s*(min|minute|min\.)\b",
    re.I,
)
RE_CONC = re.compile(
    r"\b(?:sample\s+concentrations?|injected\s*polymer\s*conc(?:entration)?|injection\s*conc(?:entration)?|conc(?:entration)?)\b\s*(?:[:=]|of|were|was)?\s*([0-9]+(?:\.\d+)?(?:\s*(?:-|to)\s*[0-9]+(?:\.\d+)?)?)\s*(g/?L|g\s*/\s*L|mg/?mL|mg\s*/\s*mL|wt%|vol%)\b",
    re.I,
)
RE_INJECT_SOL = re.compile(
    r"\b(?:injected\s*polymer\s*(?:solvent|solution)|injection\s*solvent|dissolved\s*in)\s*[:=]?\s*([A-Za-z0-9\- /,\.]+?)(?=;|,|\.|\)|\n)",
    re.I,
)
RE_INJECT_VOL = re.compile(r"\b([0-9]+(?:\.\d+)?)\s*(uL|mL)\s*(?:was\s*)?(?:injected|injection)\b", re.I)

RE_PORE = re.compile(r"\b(?:pore\s*size|pore)\s*[:=Z]?\s*([0-9]+(?:\.\d+)?)\s*(A|nm|um)\b", re.I)
RE_PARTICLE = re.compile(r"\b(?:particle\s*(?:size|diameter))\s*[:=Z]?\s*([0-9]+(?:\.\d+)?)\s*(um|nm|mm)?\b", re.I)
RE_DIM = re.compile(r"\b([0-9]{2,4})\s*[x]\s*([0-9]+(?:\.\d+)?)\s*(mm|m[m]?)\b", re.I)
RE_NUM_COLS = re.compile(r"\b(?:connected\s+in\s+series|x)\s*([0-9]+)\s*(?:columns?|cols?)\b", re.I)
RE_MANUF_LINE = re.compile(
    r"\b(?:Agilent|Waters|Jordi|Tosoh|Phenomenex|Shimadzu|Thermo(?:\s*Fisher)?|Supelco|Macherey-Nagel|Chromtech)\b.*?(?:,\s*[A-Za-z].*?)?(?=;|\.|\)|\n)",
    re.I,
)

RE_COLUMN_SENT = re.compile(r"(?:\bcolumn\b|col\.)[^\.:\n]*", re.I)

RE_SOLVENT_MIX = re.compile(
    r"\b(?:mobile\s*phase|eluent|solvent\s*mixture)\b\s*[:=]?\s*([A-Za-z0-9\- /]+?)(?:\s*,?\s*(\d+(?:\.\d+)?)\s*(wt%|vol%|%|v/v|w/w))?(?=;|\.|\)|\n)",
    re.I,
)
RE_SOLVENT_PAREN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\-/ ]+?)\s*\((\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?)\s*(wt%|vol%|%|v/v|w/w)\)"
)

RE_SOLVENT_COMPONENT = re.compile(
    r"\b(\d+(?:\.\d+)?(?:\s*(?:-|to)\s*\d+(?:\.\d+)?)?)\s*(wt%|w/w|vol%|v/v|%)\s*([A-Za-z][A-Za-z0-9\-]+)\b",
    re.I,
)
RE_SOLVENT_COMPONENT_PAREN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*%\s*\((w/w|v/v)\)\s*([A-Za-z][A-Za-z0-9\-]+)\b",
    re.I,
)

RE_SOLVENT_MIXTURE = re.compile(r"\b([A-Za-z][A-Za-z0-9]+(?:\s*-\s*[A-Za-z][A-Za-z0-9]+)+)\b")
RE_SOLVENT_MIXTURE_SLASH = re.compile(r"\b([A-Za-z][A-Za-z0-9\-]+(?:\s*/\s*[A-Za-z][A-Za-z0-9\-]+)+)\b")
RE_SOLVENT_MIXTURE_OF = re.compile(
    r"\bmixture\s+of\s+([A-Za-z][A-Za-z0-9\-]+)\s+(?:and|with)\s+([A-Za-z][A-Za-z0-9\-]+)\b",
    re.I,
)
