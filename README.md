# LCCC Condition Extraction

Extract LCCC (liquid chromatography at critical conditions) experiment conditions from raw paper text with minimal hardcoding.

## Quick Start

```powershell
python runner.py rawData/paper.txt --no-validation
```

Output is written to `extracted_lccc_data.json` and contains `LCCC_conditions`.

Example output shape:
```json
{
  "metadata": { "..." : "..." },
  "LCCC_conditions": [
    {
      "critical_polymer_unit": "EO",
      "SP_FIELDS": { "column_name": "..." },
      "SOL_FIELDS": { "solvent": "...", "ratio": "...", "ratio_units": "..." },
      "TECH_FIELDS": { "temperature": "...", "flow_rate": "..." },
      "separation_behavior": { "critical_block_behavior": "...", "purpose": "..." }
    }
  ]
}
```

## Output Mode

To keep the JSON output clean (no evidence/debug fields), use the default `compact` mode.
You can switch to `full` to keep all evidence/debug fields for troubleshooting.

```powershell
# Clean output (default)
python runner.py rawData/paper.txt --output-mode compact

# Full output with evidence/debug fields
python runner.py rawData/paper.txt --output-mode full
```

## LLM Extraction (Ollama)
The pipeline uses local Ollama models as the primary extraction engine.
All extracted values are validated against the source text.

```powershell
# Single model
python runner.py rawData/paper.txt --llm-provider ollama --llm-model qwen2.5:7b

# Multi-model with consensus
python runner.py rawData/paper.txt --llm-models "qwen3:8b,kimi-k2:7b" --llm-consensus 2
```

LLM configuration via env vars (optional):
- `LCCC_LLM_PROVIDER` (default: `ollama`)
- `LCCC_LLM_MODEL` (default: `qwen2.5:7b`)
- `LCCC_LLM_MODELS` (comma list, e.g. `qwen3:8b,kimi-k2:7b`)
- `LCCC_LLM_CONSENSUS` (min model agreement)
- `LCCC_LLM_HOST` (default: `http://127.0.0.1:11434`)
- `LCCC_LLM_TIMEOUT` (seconds)
- `LCCC_LLM_MAX_CONTEXT` (characters)
- `LCCC_LLM_DEBUG` (set to `1` for per-chunk logging)

## Project Layout

- `runner.py`: CLI runner and human-readable summary printer.
- `pipeline.py`: Compatibility wrapper (keeps the old `from pipeline import ...` import path working).
- `lccc_extractor/pipeline.py`: Pipeline orchestration (`run_pipeline`, `save_json`).
- `lccc_extractor/llm/`: LLM client + extractor.
- `lccc_extractor/polymers/`: Polymer catalog + validation.
- `lccc_extractor/text/`: Text normalization + paragraph splitting utilities.
- `lccc_extractor/linking.py`: Condition linking + validation.
- `polymerSubject.py`: Compatibility wrapper for legacy imports.

## Environment Variables

- `LCCC_LOG_LEVEL`: `DEBUG`, `INFO` (default), `WARNING`, ...
