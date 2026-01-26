import re
from collections import Counter

def parse_lccc_file(file_path):
    """Parse a text file with LCCC CONDITION blocks."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    condition_blocks = re.split(r"===== LCCC CONDITION \d+ =====", text)
    
    records = []
    for block in condition_blocks:
        if not block.strip():
            continue

        solvents = re.findall(r'\b[a-zA-Z][\w-]*/[a-zA-Z][\w-]*\b', block)

        ratios = re.findall(r'\b\d{1,3}[:/]\d{1,3}\b', block)
        
        context = block.strip()
        
        records.append({
            "solvents": solvents,
            "ratios": ratios,
            "context": context
        })
    
    return records

def find_most_common(records, min_count=1):
    """Return solvent+ratio pairs and their occurrences."""
    pairs = []
    for rec in records:
        solvents = rec.get("solvents", [])
        ratios = rec.get("ratios", [])
        if not solvents:
            solvents = ["<UNKNOWN>"]
        if not ratios:
            ratios = ["<UNKNOWN>"]

        ratios = [r.replace(":", "/") for r in ratios]
        
        for s in solvents:
            if not re.search(r'[a-zA-Z]', s):
                continue
            for r in ratios:
                pairs.append((s, r))
    
    counts = Counter(pairs)
    if not counts:
        return []
    
    max_occurrence = max(counts.values())
    most_common_pairs = [(pair, c) for pair, c in counts.items() if c == max_occurrence]
    
    return most_common_pairs

def save_most_common(most_common_pairs, save_path):
    """Save the most common solvent/ratio combinations to a file."""
    with open(save_path, "w", encoding="utf-8") as f:
        if not most_common_pairs:
            f.write("No solvent systems found.\n")
        else:
            for (solvent, ratio), count in most_common_pairs:
                f.write(f"Solvent system: {solvent}, Ratio: {ratio}, Occurrences: {count}\n")
    
    print(f"Most common solvent systems saved to {save_path}")


file_path = "LCCC_conditions.txt"          
save_path = "most_common_solvents.txt"     

records = parse_lccc_file(file_path)
most_common_pairs = find_most_common(records)
save_most_common(most_common_pairs, save_path)
