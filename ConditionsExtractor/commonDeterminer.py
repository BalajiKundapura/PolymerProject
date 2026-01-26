from collections import Counter, defaultdict
import re

RATIO_REGEX = re.compile(r"\b\d+\s*[:/]\s*\d+\b")

def extract_linked_ratios(context, polymers, solvents, window=50):
    linked_ratios = []

    for match in RATIO_REGEX.finditer(context):
        start, end = match.span()
        ratio = match.group()

        for poly in polymers:
            poly_idx = context.lower().find(poly.lower())
            if poly_idx != -1 and abs(poly_idx - start) <= window:
                linked_ratios.append(ratio)


        for sol in solvents:
            sol_idx = context.lower().find(sol.lower())
            if sol_idx != -1 and abs(sol_idx - start) <= window:
                linked_ratios.append(ratio)

    return linked_ratios

def parse_lccc_file(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    condition_blocks = [b.strip() for b in text.split("===== LCCC CONDITION ") if b.strip()]

    for block in condition_blocks:
        lines = block.splitlines()
        solvents = []
        polymers = []
        context_lines = []

        for line in lines:
            line = line.strip()
            if line.startswith("Polymers:"):
                polymer_line = line.replace("Polymers:", "").strip()
                polymer_line = polymer_line.strip("[]")
                polymers = [p.strip().strip("'\"") for p in polymer_line.split(",") if p.strip()]
            elif line.startswith("Solvent expressions:"):
                solvent_line = line.replace("Solvent expressions:", "").strip()
                solvent_line = solvent_line.strip("[]")
                solvents = [s.strip().strip("'\"") for s in solvent_line.split(",") if s.strip()]
            else:
                context_lines.append(line)

        context = " ".join(context_lines).strip()

        records.append({
            "solvents": solvents or ["<UNKNOWN>"],
            "polymers": polymers or ["<UNKNOWN>"],
            "context": context
        })

    return records

def select_best_solvent(records, alpha=2):
    solvent_count = Counter()
    solvent_polymers = defaultdict(set)
    solvent_linked_ratios = defaultdict(Counter)

    for rec in records:
        solvents = rec["solvents"]
        polymers = rec["polymers"]
        context = rec["context"]

        for s in solvents:
            if s == "<UNKNOWN>":
                continue
            solvent_count[s] += 1
            for p in polymers:
                if p != "<UNKNOWN>":
                    solvent_polymers[s].add(p)

            linked_ratios = extract_linked_ratios(context, polymers, [s])
            for r in linked_ratios:
                solvent_linked_ratios[s][r] += 1

    if not solvent_count:
        return None

    solvent_scores = {s: solvent_count[s] + alpha * len(solvent_polymers[s])
                      for s in solvent_count}

    best_solvent = max(solvent_scores, key=lambda x: solvent_scores[x])
    best_score = solvent_scores[best_solvent]

    most_common_ratio = None
    if solvent_linked_ratios.get(best_solvent):
        most_common_ratio = solvent_linked_ratios[best_solvent].most_common(1)[0][0]

    linked_polymers = sorted(list(solvent_polymers.get(best_solvent, [])))

    return {
        "solvent": best_solvent,
        "score": best_score,
        "most_common_ratio": most_common_ratio,
        "linked_polymers": linked_polymers
    }

if __name__ == "__main__":
    file_path = "LCCC_conditions.txt"
    records = parse_lccc_file(file_path)
    most_common_solvent = select_best_solvent(records)

    if most_common_solvent:
        print("Most common solvent system selected:")
        print(f"Solvent: {most_common_solvent['solvent']}")
        print(f"Score: {most_common_solvent['score']}")
        print(f"Most common linked ratio: {most_common_solvent['most_common_ratio']}")
        print(f"Linked polymers: {most_common_solvent['linked_polymers']}")
    else:
        print("⚠️ No solvent systems found.")
