# medextract

Extract blood pressure readings from medical device photos using a local AI model — no API key, no internet, no cost.

Built on [Ollama](https://ollama.com) + MedGemma, the library processes photos of BP monitors and returns structured data (systolic, diastolic, pulse, brand, AHA classification, and 10 more fields) ready for CSV export or direct use in Python.

![Tests](https://github.com/shaunakmirajgaonkar/Healthnexaa/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.3.0-blue)
![PyPI](https://img.shields.io/pypi/v/medextract)

---

## Features

- **Fully local** — runs on your machine via Ollama, no data sent to any server
- **No API key required** — free to use with no rate limits
- **14 fields extracted per image** — systolic, diastolic, pulse, brand, date, time, IHB, AFib, battery, glare, confidence, and more
- **Parallel processing** — multiple images processed simultaneously
- **Progress bar** — live tqdm progress with current reading and ETA
- **Resume support** — safely restart a crashed 1000-image batch without reprocessing
- **AHA BP classification** — Normal → Elevated → Stage 1/2 → Hypertensive Crisis
- **Validation** — flags out-of-range and physiologically impossible readings
- **Input validation** — clear errors for invalid parameters before processing starts
- **37 tests, CI on every push** — GitHub Actions runs tests across Python 3.10, 3.11, 3.12
- **Works as a Python library or CLI tool**

---

## Prerequisites

Install [Ollama](https://ollama.com), then pull the model:

```bash
ollama serve
ollama pull medgemma1.5:4b
```

---

## Install

**From PyPI:**
```bash
pip install medextract
```

**From GitHub:**
```bash
pip install git+https://github.com/shaunakmirajgaonkar/Healthnexaa.git
```

**For local development:**
```bash
git clone https://github.com/shaunakmirajgaonkar/Healthnexaa.git
cd Healthnexaa
pip install -e .
```

**Dependencies installed automatically:** `ollama>=0.6.1`, `pillow>=10.2.0`, `pandas>=2.1.1`, `tqdm>=4.66.1`

---

## Quickstart

### Process a folder

```python
from medextract import extract_folder

rows = extract_folder("/path/to/bp-monitor-photos")

for row in rows:
    print(row["systolic"], "/", row["diastolic"], "—", row["bp_classification"])
```

### Single image

```python
from medextract import analyze_image, classify_bp

result = analyze_image("/path/to/photo.jpg")

if result:
    print("Systolic :", result["systolic"])
    print("Diastolic:", result["diastolic"])
    print("Pulse    :", result["pulse"])
    print("Brand    :", result["brand"])
    print("Category :", classify_bp(result["systolic"], result["diastolic"]))
    print("Confidence:", result["confidence"], "/ 10")
```

### Save to CSV

```python
import pandas as pd
from medextract import extract_folder

rows = extract_folder("/path/to/photos")
pd.DataFrame(rows).to_csv("results.csv", index=False)
```

### Validate readings

```python
from medextract import extract_folder, validate_bp, classify_bp

rows = extract_folder("/path/to/photos")

for row in rows:
    warnings = validate_bp(row)
    category = classify_bp(row["systolic"], row["diastolic"])
    print(f"{row['file_name']:30s}  {row['systolic']}/{row['diastolic']}  {category}")
    if warnings:
        print("  WARNINGS:", warnings)
```

### 1000+ images — progress bar and resume

```python
from medextract import extract_folder

# tqdm progress bar shown automatically with live BP readout and ETA
rows = extract_folder("/path/to/1000-photos", workers=5)
```

If the run crashes midway, resume without reprocessing done images:

```python
rows = extract_folder(
    "/path/to/photos",
    resume_csv="results.csv",   # skips files already in this CSV
    workers=5,
)
```

### All parameters

```python
from medextract import extract_folder

rows = extract_folder(
    folder="/path/to/photos",
    model="medgemma1.5:4b",     # any Ollama vision model
    workers=5,                   # parallel workers (more = faster on M2/M3)
    image_size=768,              # max px before encoding (larger = more accurate)
    max_retries=5,               # retries per image on failure
    resume_csv="results.csv",    # resume a crashed batch
)
```

### CLI

```bash
# Basic
python3 -m medextract.cli /path/to/photos --output results.csv

# Full options
python3 -m medextract.cli /path/to/photos \
    --output results.csv \
    --workers 5 \
    --model medgemma1.5:4b \
    --image-size 768 \
    --max-retries 5 \
    --resume

# Help
python3 -m medextract.cli --help
```

---

## Output Fields

| Field | Type | Description |
|---|---|---|
| `file_name` | string | Source image filename |
| `systolic` | int | Top BP number (0 if unreadable) |
| `diastolic` | int | Bottom BP number (0 if unreadable) |
| `pulse` | int | Pulse / BPM (0 if unreadable) |
| `brand` | string | Device brand ("Unknown" if not visible) |
| `date` | string | Date shown on device (YYYY-MM-DD or null) |
| `time` | string | Time shown on device (HH:MM or null) |
| `memory_slot` | string | M1, M2, or null |
| `ihb` | bool | Irregular heartbeat indicator shown |
| `afib` | bool | AFib indicator shown |
| `battery_low` | bool | Battery warning shown |
| `has_glare` | bool | Glare affecting readability |
| `confidence` | int 1–10 | Model confidence (always clamped to 1–10) |
| `bp_classification` | string | AHA category |
| `extracted_at` | string | Timestamp (YYYY-MM-DD HH:MM:SS) |

---

## BP Classification (AHA Standard)

| Classification | Systolic | Diastolic |
|---|---|---|
| Normal | < 120 | < 80 |
| Elevated | 120–129 | < 80 |
| Stage 1 Hypertension | 130–139 | 80–89 |
| Stage 2 Hypertension | ≥ 140 | ≥ 90 |
| HYPERTENSIVE CRISIS | > 180 | > 120 |

---

## Performance (Apple Silicon)

| Workers | 100 images | 500 images | 1000 images |
|---|---|---|---|
| 3 (default) | ~19 min | ~97 min | ~3.2 hrs |
| 5 | ~12 min | ~58 min | ~1.9 hrs |
| 8 | ~7 min | ~36 min | ~1.2 hrs |

---

## Supported Image Formats

`.png` `.jpg` `.jpeg` `.webp`

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

37 tests covering `classify_bp`, `validate_bp`, image loading, Ollama checks, input validation, resume, and `analyze_image` — all with mocked Ollama responses (no model required to run tests).

Tests run automatically on every push and pull request via GitHub Actions across Python 3.10, 3.11, and 3.12.

---

## Project Structure

```
medextract/                  ← installable Python library
├── __init__.py              ← public API (5 exports)
├── extractor.py             ← core logic — no global state
└── cli.py                   ← command-line interface

.github/workflows/
└── tests.yml                ← CI — auto-runs tests on every push

tests/                       ← pytest test suite (37 tests)
└── test_extractor.py

examples/                    ← original standalone scripts (reference only)
├── README.md
└── *.py

plans/                       ← project documentation
├── usage-guide.md
├── library-creation-guide.md
├── project-plan.md
└── medical-device-extraction-guide.md

pyproject.toml               ← package config, pinned deps
```

---

## License

MIT — see [LICENSE](LICENSE)
