"""
Compatibility wrapper.

The LCCC extraction implementation was split into a small package under
`lccc_extractor/` so the codebase is easier to navigate and extend.

This file keeps the original public API stable for `runner.py` and any other
scripts that do:

  from pipeline import load_text, run_pipeline, save_json

"""

from lccc_extractor import load_text, run_pipeline, save_json

__all__ = ["load_text", "run_pipeline", "save_json"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract LCCC experiments with LLM-based extraction")
    ap.add_argument("input", help="Path to input text file")
    ap.add_argument("-o", "--output", default="extracted_lccc_data.json", help="Output JSON path")
    ap.add_argument("--no-validation", action="store_true", help="Skip PubChem validation")
    ap.add_argument("--threshold", type=float, default=0.5, help="Polymer selection threshold (0.0-1.0)")
    args = ap.parse_args()

    raw = load_text(args.input)
    result, metadata = run_pipeline(raw, use_validation=not args.no_validation, threshold_ratio=args.threshold)
    save_json(result, args.output, metadata)
