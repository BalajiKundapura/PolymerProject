# LCCC Condition Extraction

Extract LCCC (liquid chromatography at critical conditions) experiment conditions from raw paper text with minimal hardcoding.

## Quick Start

```powershell
python runner.py rawData/paper.txt --no-validation
```

Output is written to `extracted_lccc_data.json`.

## Accuracy + Completion Modes

The pipeline supports a completion pass that tries to fill missing fields without inventing data.

```powershell
# Conservative (no cross-record filling)
python runner.py rawData/paper.txt --completion-mode strict

# Balanced (safe fills from same column/solvent across the paper)
python runner.py rawData/paper.txt --completion-mode balanced

# Aggressive (includes optional LLM fallback if available)
python runner.py rawData/paper.txt --completion-mode aggressive
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

### Optional Local LLM (corner cases)
If you have a lightweight local model via Ollama, you can enable an LLM fallback for tricky cases.
The LLM output is only accepted when values are supported by the source text.

```powershell
# Example (model must already be pulled in Ollama)
python runner.py rawData/paper.txt --completion-mode aggressive --llm-provider ollama --llm-model phi3:mini
```

LLM configuration via env vars (optional):
- `LCCC_LLM_PROVIDER` (default: `ollama`)
- `LCCC_LLM_MODEL` (default: `phi3:mini`)
- `LCCC_LLM_TIMEOUT` (seconds)
- `LCCC_LLM_MAX_CONTEXT` (characters)

## Project Layout

- `runner.py`: CLI runner and human-readable summary printer.
- `pipeline.py`: Compatibility wrapper (keeps the old `from pipeline import ...` import path working).
- `lccc_extractor/pipeline.py`: Pipeline orchestration (`run_pipeline`, `save_json`).
- `lccc_extractor/context.py`: Paragraph classification, global-method context extraction, experiment linking.
- `lccc_extractor/extractors.py`: Regex-based extractors (stationary phase, solvents, technical details).
- `lccc_extractor/polymer_mentions.py`: Polymer mention detection (uses abbreviations from `polymerSubject.py`).
- `polymerSubject.py`: Polymer extraction + optional PubChem validation (validation gracefully skips if `aiohttp` isn't installed).
- `pipeline_monolith.py`: The pre-split monolithic pipeline kept for reference.

## Environment Variables

- `LCCC_LOG_LEVEL`: `DEBUG`, `INFO` (default), `WARNING`, ...
- `LCCC_USE_SEMANTIC`: set to `1` to try `sentence-transformers` (optional dependency).
- `LCCC_SENT_MODEL`: override sentence-transformers model name (default: `all-MiniLM-L6-v2`).
