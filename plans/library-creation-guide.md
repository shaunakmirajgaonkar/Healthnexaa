# Guide: Converting the BP Monitor Script into a Python Library

## Overview

This document describes how to convert the existing `medical_image_extraction_new.py` script into a
proper, installable Python library called `medextract`. Once packaged, anyone can install it with
`pip install medextract` and use it in their own code.

---

## Current Problems with the Script

| Problem | Why It Matters |
|---|---|
| Global constants (`IMAGE_FOLDER`, `MODEL`, etc.) | Callers cannot configure these without editing the file |
| `logging.basicConfig()` at module level | Hijacks the caller's logging setup — a library must never do this |
| No `__init__.py` | Python cannot import it as a module |
| No `pyproject.toml` | Cannot be installed with pip |
| All logic lives in `main()` | Nothing is reusable; callers can only run the whole pipeline |

---

## Target Package Structure

```
medextract/                  ← the installable package
├── __init__.py              ← public API (what users import)
├── extractor.py             ← core logic, no global state
└── cli.py                   ← command-line entry point only

pyproject.toml               ← packaging config (replaces setup.py)
```

Everything outside the `medextract/` folder (existing scripts, CSV files, plans) stays as-is.

---

## Step 1 — Create `medextract/extractor.py`

This file holds all reusable logic. Key changes from the current script:

- No module-level constants. Everything becomes a function parameter with a sensible default.
- Replace `logging.basicConfig()` with `logging.getLogger(__name__).addHandler(logging.NullHandler())`.
  This is the standard library convention: let the caller configure logging.
- Expose a single high-level function `extract_folder()` that the caller can use end-to-end.

```python
"""
medextract/extractor.py
Core extraction logic — no global state, no logging configuration.
"""

import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
from pathlib import Path

import ollama
from PIL import Image, UnidentifiedImageError

logging.getLogger(__name__).addHandler(logging.NullHandler())
log = logging.getLogger(__name__)

EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

PROMPT = """You are a medical device reader. Analyze this blood pressure monitor image and extract
every visible value. Return ONLY a valid JSON object — no markdown, no extra text:
{
  "systolic":    <int, 0 if unreadable>,
  "diastolic":   <int, 0 if unreadable>,
  "pulse":       <int, 0 if unreadable>,
  "brand":       "<string, Unknown if not visible>",
  "date":        "<YYYY-MM-DD or null>",
  "time":        "<HH:MM or null>",
  "memory_slot": "<M1, M2, or null>",
  "ihb":         <true if irregular heartbeat symbol shown, else false>,
  "afib":        <true if afib indicator shown, else false>,
  "error_code":  "<E1-E5 string or null>",
  "user":        "<User 1, User 2, or null>",
  "battery_low": <true if battery warning shown, else false>,
  "has_glare":   <true if glare affects readability, else false>,
  "confidence":  <int 1-10, your confidence in the reading>
}"""


def load_image_b64(image_path: Path, image_size: int = 512) -> str | None:
    try:
        img = Image.open(image_path)
        img.thumbnail((image_size, image_size), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except (UnidentifiedImageError, Exception) as e:
        log.warning("Cannot open %s: %s", image_path.name, e)
        return None


def analyze_image(
    image_path: Path,
    model: str = "medgemma1.5:4b",
    image_size: int = 512,
    max_retries: int = 3,
) -> dict | None:
    """Extract BP readings from a single image. Returns a dict or None on failure."""
    b64 = load_image_b64(image_path, image_size)
    if b64 is None:
        return None

    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": PROMPT, "images": [b64]}],
            )
            raw = response["message"]["content"].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            start, end = raw.find("{"), raw.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            data = json.loads(raw[start:end])
            data["file_name"] = image_path.name
            return data
        except Exception as e:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, image_path.name, e)
            if attempt < max_retries:
                time.sleep(2)

    log.error("Skipping %s: all retries exhausted.", image_path.name)
    return None


def classify_bp(systolic: int, diastolic: int) -> str:
    """Return an AHA BP category string."""
    if systolic > 180 or diastolic > 120:
        return "HYPERTENSIVE CRISIS"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension"
    elif systolic >= 130 or diastolic >= 80:
        return "Stage 1 Hypertension"
    elif systolic >= 120 and diastolic < 80:
        return "Elevated"
    elif systolic > 0:
        return "Normal"
    return "Unknown"


def validate_bp(row: dict) -> list[str]:
    """Return a list of human-readable warning strings for out-of-range values."""
    warnings = []
    s, d, p = row.get("systolic", 0), row.get("diastolic", 0), row.get("pulse", 0)
    if s and not (60 <= s <= 300):
        warnings.append(f"systolic {s} out of range")
    if d and not (30 <= d <= 200):
        warnings.append(f"diastolic {d} out of range")
    if p and not (30 <= p <= 250):
        warnings.append(f"pulse {p} out of range")
    if s and d and s <= d:
        warnings.append("systolic must be greater than diastolic")
    return warnings


def extract_folder(
    folder: str | Path,
    model: str = "medgemma1.5:4b",
    workers: int = 3,
    image_size: int = 512,
    max_retries: int = 3,
) -> list[dict]:
    """
    Process every image in `folder` and return a list of result dicts.

    Each dict contains the extracted BP fields plus:
      - bp_classification: AHA category string
      - extracted_at: ISO timestamp string
    """
    folder = Path(folder)
    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in EXTENSIONS)
    if not images:
        log.warning("No images found in %s", folder)
        return []

    results: dict[int, dict] = {}

    def _process(args):
        idx, path = args
        result = analyze_image(path, model=model, image_size=image_size, max_retries=max_retries)
        return idx, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, (i, img)): i for i, img in enumerate(images)}
        for future in as_completed(futures):
            idx, result = future.result()
            if result:
                result["bp_classification"] = classify_bp(
                    result.get("systolic", 0), result.get("diastolic", 0)
                )
                result["extracted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                results[idx] = result

    return [results[i] for i in sorted(results)]
```

---

## Step 2 — Create `medextract/__init__.py`

This controls what users see when they do `from medextract import ...`.

```python
"""medextract — extract readings from medical device images using a local LLM."""

from .extractor import analyze_image, classify_bp, extract_folder, validate_bp

__all__ = ["extract_folder", "analyze_image", "classify_bp", "validate_bp"]
```

---

## Step 3 — Create `medextract/cli.py`

The CLI is a thin wrapper. It parses arguments and calls `extract_folder()`.

```python
"""medextract/cli.py — command-line entry point."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .extractor import extract_folder

COLUMN_ORDER = [
    "file_name", "systolic", "diastolic", "pulse", "brand",
    "date", "time", "memory_slot", "ihb", "afib",
    "error_code", "user", "battery_low", "has_glare", "confidence",
    "bp_classification", "extracted_at",
]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="medextract",
        description="Extract readings from BP monitor images using a local LLM.",
    )
    parser.add_argument("folder", help="Path to folder containing images")
    parser.add_argument("--output", default="medical_data_export.csv", help="Output CSV path")
    parser.add_argument("--model", default="medgemma1.5:4b")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    rows = extract_folder(
        args.folder,
        model=args.model,
        workers=args.workers,
        image_size=args.image_size,
        max_retries=args.max_retries,
    )

    if not rows:
        print("No data extracted.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df[[c for c in COLUMN_ORDER if c in df.columns]]
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
```

---

## Step 4 — Create `pyproject.toml`

Place this in the repo root (same level as `medextract/`).

> **Note:** Use `setuptools>=42` with `setuptools.build_meta` — the newer `setuptools.backends.legacy:build` requires a very recent setuptools version and will fail on most machines.

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "medextract"
version = "0.1.0"
description = "Extract readings from medical device images using a local LLM (Ollama + MedGemma)"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
keywords = ["medical", "blood-pressure", "ocr", "llm", "ollama"]
dependencies = [
    "ollama",
    "pillow",
    "pandas",
]

[project.scripts]
medextract = "medextract.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["medextract*"]
```

---

## Step 5 — Install and Test

### Install locally (editable mode — changes take effect immediately)

```bash
pip install -e .
```

### Test as a Python import

```python
from medextract import extract_folder

rows = extract_folder("/path/to/images", model="medgemma1.5:4b", workers=4)
for row in rows:
    print(row["systolic"], row["diastolic"], row["bp_classification"])
```

### Test the CLI

```bash
medextract /path/to/images --output results.csv --workers 4
```

---

## Step 6 — Publish to PyPI (optional, for public distribution)

```bash
# Install build tools
pip install build twine

# Build the distribution
python -m build

# Upload to PyPI (you need a PyPI account and API token)
twine upload dist/*
```

After publishing, anyone can install it with:

```bash
pip install medextract
```

---

## Summary of Changes from Script to Library

| Before (script) | After (library) |
|---|---|
| Global `IMAGE_FOLDER`, `MODEL` constants | Parameters with defaults on `extract_folder()` |
| `logging.basicConfig()` at top level | `NullHandler` in library; `basicConfig` only in `cli.py` |
| Single flat `.py` file | `medextract/` package with 3 focused modules |
| No install path | `pip install -e .` or `pip install medextract` |
| Run with `python script.py` | Import in code or run `python3 -m medextract.cli` |

---

## Status — IMPLEMENTED ✅ (2026-04-25)

All files have been created and the library is installed and verified working.

**Confirmed working on:**
- Python 3.12, macOS Darwin 25.4.0
- Ollama + `medgemma1.5:4b` (3.3 GB local model)
- Real Omron and Meditech BP monitor photos

**Sample output from real images:**

```
Image: Unknown-2.jpeg (Meditech BP-12)
  Systolic : 130
  Diastolic: 80
  Pulse    : 76
  Brand    : MEDITECH
  Confidence: 10 / 10
  Category : Stage 1 Hypertension

Image: 20210805_..._F_3000_4000.jpg (Omron)
  Systolic : 140
  Diastolic: 80
  Pulse    : 70
  Brand    : Omron
  Confidence: 10 / 10
  Category : Stage 2 Hypertension
```
