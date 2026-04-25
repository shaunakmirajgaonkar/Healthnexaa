# Guide: Converting the BP Monitor Script into a Python Library

**Version:** 0.4.0 | **Status:** Production Ready | **Tests:** 37 passing

## Overview

This document describes how the `medextract` library was built — from a standalone script to a
fully installable, tested Python package. The library is complete and production-ready.

Anyone can install it with `pip install git+https://github.com/shaunakmirajgaonkar/Healthnexaa.git`
and use it in their own code.

---

## Problems the Original Script Had

| Problem | Why It Matters |
|---|---|
| Global constants (`IMAGE_FOLDER`, `MODEL`, etc.) | Callers cannot configure these without editing the file |
| `logging.basicConfig()` at module level | Hijacks the caller's logging setup — a library must never do this |
| No `__init__.py` | Python cannot import it as a module |
| No `pyproject.toml` | Cannot be installed with pip |
| All logic lives in `main()` | Nothing is reusable; callers can only run the whole pipeline |
| No tests | No way to verify correctness without running against real devices |
| No input validation | Bad parameters cause cryptic errors deep in the stack |

---

## Final Package Structure

```
medextract/                  ← installable Python library
├── __init__.py              ← public API (5 exports)
├── extractor.py             ← core logic — no global state
└── cli.py                   ← command-line interface

.github/workflows/
└── tests.yml                ← CI — auto-runs tests on every push

tests/
└── test_extractor.py        ← 37 pytest tests, Ollama fully mocked

examples/                    ← original standalone scripts (reference only)
├── README.md
└── *.py

plans/                       ← project documentation
pyproject.toml               ← package config, pinned deps
```

---

## Step 1 — `medextract/extractor.py`

Core logic. No global state. Everything is a function parameter with a sensible default.

Key rules enforced:
- `NullHandler` only — caller controls logging, library never calls `basicConfig()`
- All public functions validate parameters and raise `ValueError` or `RuntimeError` with clear messages
- Confidence always clamped to 1–10 before returning
- Resume support via `already_done` set built from existing CSV

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
import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

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


def check_ollama(model: str = "medgemma1.5:4b") -> None:
    """Raise RuntimeError if Ollama is not running or the model is not pulled."""
    try:
        pulled = [m["model"] for m in ollama.list()["models"]]
    except Exception:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve"
        )
    if model not in pulled:
        raise RuntimeError(
            f"Model '{model}' is not pulled. Run: ollama pull {model}"
        )


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
            data["confidence"] = max(1, min(10, int(data.get("confidence", 5))))
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
    resume_csv: str | None = None,
) -> list[dict]:
    """
    Process every image in `folder` and return a list of result dicts.

    Raises ValueError for invalid parameters.
    Raises RuntimeError if Ollama is not running or model is not pulled.
    """
    # Input validation — fail fast with clear messages
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if image_size < 64:
        raise ValueError(f"image_size must be >= 64, got {image_size}")
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")

    check_ollama(model)

    folder = Path(folder)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder}")

    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in EXTENSIONS)
    if not images:
        log.warning("No images found in %s", folder)
        return []

    # Resume: skip images already in existing CSV
    already_done: set[str] = set()
    existing_rows: list[dict] = []
    if resume_csv is not None:
        resume_path = Path(resume_csv)
        if resume_path.exists():
            existing_df = pd.read_csv(resume_path)
            already_done = set(existing_df["file_name"].tolist())
            existing_rows = existing_df.to_dict("records")

    images = [img for img in images if img.name not in already_done]
    if not images:
        log.info("All images already processed (resume_csv).")
        return existing_rows

    results: dict[int, dict] = {}

    def _process(args):
        idx, path = args
        result = analyze_image(path, model=model, image_size=image_size, max_retries=max_retries)
        return idx, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, (i, img)): i for i, img in enumerate(images)}
        with tqdm(total=len(images), unit="img") as bar:
            for future in as_completed(futures):
                idx, result = future.result()
                if result:
                    result["bp_classification"] = classify_bp(
                        result.get("systolic", 0), result.get("diastolic", 0)
                    )
                    result["extracted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    results[idx] = result
                    bar.set_postfix(
                        bp=f"{result.get('systolic', 0)}/{result.get('diastolic', 0)}"
                    )
                bar.update(1)

    new_rows = [results[i] for i in sorted(results)]
    return existing_rows + new_rows
```

---

## Step 2 — `medextract/__init__.py`

```python
"""medextract — extract readings from medical device images using a local LLM."""

from .extractor import analyze_image, check_ollama, classify_bp, extract_folder, validate_bp

__version__ = "0.3.0"
__all__ = ["extract_folder", "analyze_image", "classify_bp", "validate_bp", "check_ollama"]
```

---

## Step 3 — `medextract/cli.py`

The CLI is a thin wrapper. All logic stays in `extractor.py`.

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
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already in the output CSV")
    args = parser.parse_args()

    rows = extract_folder(
        args.folder,
        model=args.model,
        workers=args.workers,
        image_size=args.image_size,
        max_retries=args.max_retries,
        resume_csv=args.output if args.resume else None,
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

## Step 4 — `pyproject.toml`

> **Important:** Use `setuptools>=42` with `setuptools.build_meta` — the newer
> `setuptools.backends.legacy:build` requires a very recent setuptools version and will fail on
> many machines.

```toml
[build-system]
requires = ["setuptools>=42"]
build-backend = "setuptools.build_meta"

[project]
name = "medextract"
version = "0.3.0"
description = "Extract readings from medical device images using a local LLM (Ollama + MedGemma)"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
keywords = ["medical", "blood-pressure", "ocr", "llm", "ollama"]
dependencies = [
    "ollama>=0.6.1",
    "pillow>=10.2.0",
    "pandas>=2.1.1",
    "tqdm>=4.66.1",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-mock>=3.12"]

[project.scripts]
medextract = "medextract.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["medextract*"]
```

---

## Step 5 — `tests/test_extractor.py`

37 tests covering all public functions. Ollama is fully mocked — tests run without a model.

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Test groups:

| Class | Tests | What It Covers |
|---|---|---|
| `TestClassifyBP` | 10 | All AHA categories, edge values |
| `TestValidateBP` | 8 | Out-of-range, inversion, boundary values |
| `TestLoadImageB64` | 5 | Image load, resize, bad file, non-RGB |
| `TestCheckOllama` | 3 | Running, not running, model missing |
| `TestAnalyzeImage` | 5 | Success, retry, markdown stripping, confidence clamp |
| `TestExtractFolderValidation` | 4 | workers/image_size/max_retries ValueError, missing folder |
| `TestExtractFolderResume` | 2 | Skips already-done, returns combined rows |

---

## Step 6 — `.github/workflows/tests.yml` (CI)

GitHub Actions runs the full test suite on every push and pull request across Python 3.10, 3.11, and 3.12.

```yaml
name: Tests
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
```

---

## Step 7 — Install and Test

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
python3 -m medextract.cli /path/to/images --output results.csv --workers 4
```

### Run the test suite

```bash
pytest tests/ -v
```

---

## Step 8 — Publish to PyPI ✅ (2026-04-25)

```bash
pip install build twine
python3 -m build
python3 -m twine upload dist/* --username __token__ --password <your-pypi-token>
```

**Live on PyPI:** https://pypi.org/project/medextract/0.3.0/

Anyone can now install it with:

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
| No install path | `pip install medextract` (PyPI) or `pip install -e .` (local) |
| Run with `python script.py` | Import in code or run `python3 -m medextract.cli` |
| No tests | 37 pytest tests — all passing, Ollama fully mocked |
| No input validation | `ValueError` with clear message before processing starts |
| No Ollama check | `check_ollama()` raises `RuntimeError` with fix instructions |
| No progress feedback | tqdm progress bar with live BP readout and ETA |
| No resume | `resume_csv` parameter skips already-processed images |
| No CI | GitHub Actions across Python 3.10, 3.11, 3.12 |

---

## Status — PUBLISHED ON PyPI ✅ (2026-04-25)

All files created, tested, verified, and published to PyPI.

**Install:** `pip install medextract`
**PyPI:** https://pypi.org/project/medextract/0.3.0/
**GitHub:** https://github.com/shaunakmirajgaonkar/Healthnexaa

**Verified on:**
- Python 3.12, macOS Darwin 25.4.0
- Ollama + `medgemma1.5:4b` (3.3 GB local model)
- Real Omron and Meditech BP-12 photos

**Verified test results:**

| Image | Device | BP | Pulse | Brand | Confidence | Category |
|---|---|---|---|---|---|---|
| Unknown-2.jpeg | Meditech BP-12 | 130/80 | 76 | MEDITECH | 10/10 | Stage 1 Hypertension |
| 20210805_..._F_3000_4000.jpg | Omron | 140/80 | 70 | Omron | 10/10 | Stage 2 Hypertension |
| Unknown-9.jpeg | Unknown (glare) | 0/0 | 0 | Unknown | 5/10 | Unknown (handled gracefully) |

**Automated tests:** 37 / 37 PASS

**CI:** GitHub Actions passing on Python 3.10, 3.11, 3.12
