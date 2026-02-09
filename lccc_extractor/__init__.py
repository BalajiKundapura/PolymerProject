"""
LCCC Condition Extraction package.

This package contains the extraction pipeline and supporting utilities.
The main entrypoints are:
  - `run_pipeline(text, ...)`
  - `load_text(path)`
  - `save_json(data, out_path, metadata)`

For a CLI, use `runner.py` in the repo root.
"""

from .pipeline import run_pipeline, save_json
from .text_utils import load_text

__all__ = ["load_text", "run_pipeline", "save_json"]

