import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List

from pipeline import load_text, run_pipeline, save_json
from lccc_extractor.llm import LLMConfig
from lccc_extractor.reporting import write_conditions_csv
from lccc_extractor.postprocess import llm_normalize_conditions_table

def print_summary(data: List[Dict[str, Any]], metadata: Dict[str, Any]) -> None:
    """Print extraction summary with context information"""
    def _safe(text: str) -> str:
        return "".join(ch if ord(ch) < 128 else "?" for ch in text)

    print("\n" + "=" * 100)
    print("LCCC EXTRACTION SUMMARY - LLM-BASED ANALYSIS")
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

    print("\nLCCC CONDITIONS:")
    print(f"  Extracted conditions:            {metadata.get('extracted_conditions', 0)}")

    llm_meta = metadata.get("llm_extraction") or {}
    if llm_meta:
        print("\nLLM EXTRACTION:")
        models = llm_meta.get("models") or []
        model_display = ", ".join(models) if models else llm_meta.get("model", "")
        print(f"  Model(s):                        {model_display}")
        print(f"  Available:                       {llm_meta.get('llm_available', False)}")
        print(f"  Chunks:                          {llm_meta.get('chunks', 0)}")
        print(f"  LLM calls:                       {llm_meta.get('llm_calls', 0)}")
        print(f"  Conditions from LLM:             {llm_meta.get('conditions', 0)}")
        if llm_meta.get("consensus_min"):
            print(f"  Consensus min:                   {llm_meta.get('consensus_min')}")

    post_meta = metadata.get("llm_postprocess") or {}
    if post_meta:
        print("\nLLM POSTPROCESS:")
        print(f"  Used:                            {post_meta.get('used', False)}")
        if post_meta.get("used"):
            if post_meta.get("model"):
                print(f"  Model:                           {post_meta.get('model')}")
            if post_meta.get("row_count") is not None:
                print(f"  Rows after cleanup:              {post_meta.get('row_count')}")
        else:
            if post_meta.get("reason"):
                print(f"  Reason:                          {post_meta.get('reason')}")

    total_conditions = len(data) if isinstance(data, list) else 0
    print(f"  Total conditions:                {total_conditions}")

    if total_conditions > 0:
        print("\n" + "-" * 100)
        print("DETAILED CONDITIONS:")
        print("-" * 100)

        for idx, cond in enumerate(data, start=1):
            print(f"\n  Condition {idx}:")

            unit = cond.get("critical_polymer_unit")
            if unit:
                print(f"    Critical unit: {unit}")

            sp = cond.get("SP_FIELDS") or {}
            if sp:
                details = [f"{k}: {v}" for k, v in sp.items() if v is not None]
                if details:
                    print("    Stationary Phase:")
                    for detail in details:
                        print(f"      - {detail}")

            sol = cond.get("SOL_FIELDS") or {}
            if sol.get("solvent"):
                ratio_str = f" ({sol.get('ratio')} {sol.get('ratio_units')})" if sol.get("ratio") else ""
                print(f"    Solvent: {sol.get('solvent')}{ratio_str}")

            tech = cond.get("TECH_FIELDS") or {}
            tech_details = [f"{k}: {v}" for k, v in tech.items() if v is not None]
            if tech_details:
                print("    Technical Details:")
                for detail in tech_details:
                    print(f"      - {detail}")

            behavior = cond.get("separation_behavior") or {}
            beh_details = [f"{k}: {v}" for k, v in behavior.items() if v is not None]
            if beh_details:
                print("    Separation Behavior:")
                for detail in beh_details:
                    print(f"      - {detail}")
    else:
        print("\nNo LCCC conditions found")

    print(f"\nPROCESSING TIME: {metadata.get('processing_time_seconds', 0):.2f}s")
    print("=" * 100 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Extract LCCC experimental data with LLM-based extraction",
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
        "--output-mode",
        choices=["compact", "full"],
        default="compact",
        help="Output schema mode (compact removes evidence/debug fields)",
    )
    parser.add_argument("--llm-provider", default=None, help="Local LLM provider (e.g., ollama)")
    parser.add_argument("--llm-model", default=None, help="Local LLM model name (e.g., phi3:mini)")
    parser.add_argument("--llm-models", default=None, help="Comma-separated list of LLM models (e.g., qwen3:8b,kimi-k2:7b)")
    parser.add_argument("--llm-consensus", type=int, default=None, help="Min model agreement to keep a condition (default: 2 if multiple models)")
    parser.add_argument("--llm-host", default=None, help="Ollama host URL (default: http://127.0.0.1:11434)")
    parser.add_argument("--llm-no-start-server", action="store_true", help="Do not try to start ollama server automatically")
    parser.add_argument("--llm-no-http", action="store_true", help="Disable Ollama HTTP API (use CLI only)")
    parser.add_argument("--llm-timeout", type=int, default=None, help="LLM timeout seconds")
    parser.add_argument("--llm-max-context", type=int, default=None, help="Max chars to send to LLM")
    parser.add_argument("--no-llm-postprocess", action="store_true", help="Disable LLM postprocess table cleanup")
    parser.add_argument("--llm-postprocess-max-rows", type=int, default=60, help="Max table rows for LLM postprocess")
    parser.add_argument("--table-csv", default=None, help="Optional path to save tabular CSV output")
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
        print("Running LLM-based LCCC extraction pipeline...")
        print(f"   (validation={'enabled' if not args.no_validation else 'disabled'}, threshold={args.threshold})\n")
        
        llm_config = None
        if any([
            args.llm_provider,
            args.llm_model,
            args.llm_models,
            args.llm_consensus is not None,
            args.llm_host,
            args.llm_no_start_server,
            args.llm_no_http,
            args.llm_timeout,
            args.llm_max_context,
        ]):
            models = []
            if args.llm_models:
                models = [m.strip() for m in args.llm_models.split(",") if m.strip()]
            llm_config = LLMConfig(
                provider=args.llm_provider or "ollama",
                model=args.llm_model or os.getenv("LCCC_LLM_MODEL", "qwen2.5:7b"),
                models=models,
                timeout_s=args.llm_timeout or int(os.getenv("LCCC_LLM_TIMEOUT", "60")),
                max_context_chars=args.llm_max_context or int(os.getenv("LCCC_LLM_MAX_CONTEXT", "1600")),
                consensus_min=args.llm_consensus or int(os.getenv("LCCC_LLM_CONSENSUS", "0") or 0),
                host=args.llm_host or os.getenv("LCCC_LLM_HOST", os.getenv("OLLAMA_HOST", "")),
                start_server=not args.llm_no_start_server,
                use_http=not args.llm_no_http,
            )
        data, metadata, tables = run_pipeline(
            text,
            use_validation=not args.no_validation,
            threshold_ratio=args.threshold,
            llm_config=llm_config,
            output_mode=args.output_mode,
        )

        # Optional LLM postprocess: normalize/dedupe table rows
        if not args.no_llm_postprocess:
            raw_rows = (tables or {}).get("conditions", [])
            cleaned_rows, post_meta = llm_normalize_conditions_table(
                raw_rows,
                llm_config=llm_config,
                max_rows=args.llm_postprocess_max_rows,
            )
            metadata["llm_postprocess"] = post_meta
            if post_meta.get("used"):
                tables["conditions_raw"] = raw_rows
                tables["conditions"] = cleaned_rows

        # Save to JSON
        save_json(data, args.output, metadata, tables)
        if args.table_csv:
            write_conditions_csv(tables.get("conditions", []), args.table_csv)

        # Summary
        if not args.no_summary:
            print_summary(data, metadata)

        if args.table_csv:
            print(f"Tabular CSV saved to: {args.table_csv}")
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
