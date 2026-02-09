import os
import sys
import json
import time
import argparse
from typing import Dict, Any

from pipeline import load_text, run_pipeline, save_json
from lccc_extractor.llm_fallback import LLMConfig

def print_summary(data: Dict[str, Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    """Print extraction summary with context information"""
    def _safe(text: str) -> str:
        return "".join(ch if ord(ch) < 128 else "?" for ch in text)

    print("\n" + "=" * 100)
    print("LCCC EXTRACTION SUMMARY - CONTEXT-AWARE ANALYSIS")
    print("=" * 100)

    print("\nPOLYMER DETECTION:")
    print(f"  Total polymers found in text:    {metadata.get('total_polymers_found', 0)}")
    print(f"  Validated polymers (PubChem):    {metadata.get('validated_polymers', 0)}")
    print(f"  Main polymers selected:          {metadata.get('main_polymers', 0)}")

    if metadata.get('main_polymers_list'):
        print("\n  Main polymers:")
        for poly in metadata['main_polymers_list']:
            print(f"    - {poly}")

    print("\nTEXT PROCESSING:")
    print(f"  Total paragraphs:                {metadata.get('total_paragraphs', 0)}")
    print(f"  LCCC-relevant paragraphs:        {metadata.get('lccc_paragraphs', 0)}")
    print(f"  Polymer mentions found:          {metadata.get('polymer_mentions_found', 0)}")

    print("\nLCCC EXPERIMENTS:")
    print(f"  Extracted experiments:           {metadata.get('extracted_experiments', 0)}")

    completion = metadata.get("completion") or {}
    if completion:
        print("\nCOMPLETION:")
        print(f"  Mode:                            {completion.get('completion_mode', 'balanced')}")
        print(f"  Filled fields:                   {completion.get('filled_fields', 0)}")
        print(f"  LLM calls:                       {completion.get('llm_calls', 0)}")

    total_polymers = len(data)
    total_experiments = sum(len(exps) for exps in data.values())
    total_columns = sum(len(exp.get('stationary_phase', [])) for exps in data.values() for exp in exps.values())
    total_solvents = sum(len(exp.get('solvent_details', [])) for exps in data.values() for exp in exps.values())
    total_technical = sum(len(exp.get('technical_details', [])) for exps in data.values() for exp in exps.values())

    print(f"  Polymers with LCCC data:          {total_polymers}")
    print(f"  Total experiments:               {total_experiments}")
    print(f"  Stationary phases:               {total_columns}")
    print(f"  Solvent details:                 {total_solvents}")
    print(f"  Technical details:               {total_technical}")

    if total_polymers > 0:
        print("\n" + "-" * 100)
        print("DETAILED EXPERIMENTS BY POLYMER:")
        print("-" * 100)

        for poly, exps in data.items():
            print(f"\n{poly.upper()}")
            print(f"  Experiments: {len(exps)}")

            for exp_id, exp in exps.items():
                print(f"\n  {exp_id}:")

                if exp.get('context_snippet'):
                    snippet = _safe(exp['context_snippet'][:150])
                    print(f"    Context: ...{snippet}...")

                if exp.get('polymer_mention'):
                    print(f"    Polymer mention: '{_safe(exp['polymer_mention'])}'")

                sp_list = exp.get('stationary_phase', [])
                if sp_list:
                    print(f"    Stationary Phase ({len(sp_list)}):")
                    for sp in sp_list:
                        details = [f"{k}: {v}" for k, v in sp.items() if v is not None]
                        if details:
                            for detail in details:
                                print(f"      - {detail}")
                        else:
                            print("      - (no specific details)")

                sol_list = exp.get('solvent_details', [])
                if sol_list:
                    print(f"    Solvents ({len(sol_list)}):")
                    for sol in sol_list:
                        if sol.get('solvent'):
                            ratio_str = f" ({sol['ratio']} {sol['ratio_units']})" if sol.get('ratio') else ""
                            print(f"      - {sol['solvent']}{ratio_str}")

                tech_list = exp.get('technical_details', [])
                if tech_list:
                    print(f"    Technical Details ({len(tech_list)}):")
                    for tech in tech_list:
                        details = [f"{k}: {v}" for k, v in tech.items() if v is not None]
                        if details:
                            for detail in details:
                                print(f"      - {detail}")
    else:
        print("\nNo LCCC experiments found for main polymers")

    print(f"\nPROCESSING TIME: {metadata.get('processing_time_seconds', 0):.2f}s")
    print("=" * 100 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extract LCCC experimental data with context-aware polymer detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python runner.py rawData/paper.txt
  python runner.py rawData/paper.txt -o output/results.json
  python runner.py rawData/paper.txt --no-validation --threshold 0.3
        """
    )
    parser.add_argument("input", nargs="?", default="rawData/paper.txt", help="Path to input .txt file")
    parser.add_argument("-o", "--output", default="extracted_lccc_data.json", help="Output JSON file path")
    parser.add_argument("--no-validation", action="store_true", help="Skip PubChem validation (faster)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Polymer selection threshold (0.0-1.0, default 0.5)")
    parser.add_argument("--no-summary", action="store_true", help="Disable summary output")
    parser.add_argument(
        "--completion-mode",
        choices=["strict", "balanced", "aggressive"],
        default="balanced",
        help="Completion strategy for missing fields (strict|balanced|aggressive)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["compact", "full"],
        default="compact",
        help="Output schema mode (compact removes evidence/debug fields)",
    )
    parser.add_argument("--llm-provider", default=None, help="Local LLM provider (e.g., ollama)")
    parser.add_argument("--llm-model", default=None, help="Local LLM model name (e.g., phi3:mini)")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM timeout seconds")
    parser.add_argument("--llm-max-context", type=int, default=None, help="Max chars to send to LLM")
    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not 0.0 <= args.threshold <= 1.0:
        print(f"Error: Threshold must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)

    try:
        # Load
        print(f"Loading: {args.input}")
        text = load_text(args.input)
        print(f"Loaded {len(text)} characters\n")

        # Run pipeline
        print("Running context-aware LCCC extraction pipeline...")
        print(f"   (validation={'enabled' if not args.no_validation else 'disabled'}, threshold={args.threshold})\n")
        
        llm_config = None
        if any([args.llm_provider, args.llm_model, args.llm_timeout, args.llm_max_context]):
            llm_config = LLMConfig(
                provider=args.llm_provider or "ollama",
                model=args.llm_model or os.getenv("LCCC_LLM_MODEL", "phi3:mini"),
                timeout_s=args.llm_timeout or int(os.getenv("LCCC_LLM_TIMEOUT", "30")),
                max_context_chars=args.llm_max_context or int(os.getenv("LCCC_LLM_MAX_CONTEXT", "1600")),
            )
            if args.completion_mode == "balanced":
                args.completion_mode = "aggressive"

        data, metadata = run_pipeline(
            text,
            use_validation=not args.no_validation,
            threshold_ratio=args.threshold,
            completion_mode=args.completion_mode,
            llm_config=llm_config,
            output_mode=args.output_mode,
        )

        # Save to JSON
        save_json(data, args.output, metadata)

        # Summary
        if not args.no_summary:
            print_summary(data, metadata)

        print(f"Complete! Results saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"File error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
