# medextract — Usage Guide

**Version:** 0.3.0 | **Status:** Production Ready | **Tests:** 37 passing

How to install and use the `medextract` library to extract blood pressure readings from medical device images.

---

## Prerequisites (one-time setup)

The library uses **Ollama** to run the AI model locally — no API key, no internet, no cost.

```bash
# 1. Install Ollama from https://ollama.com
# 2. Start the Ollama server
ollama serve

# 3. Pull the medical vision model (3.3 GB, one-time download)
ollama pull medgemma1.5:4b
```

---

## Install the Library

**From GitHub:**
```bash
pip install git+https://github.com/shaunakmirajgaonkar/Healthnexaa.git
```

**From PyPI (recommended):**
```bash
pip install medextract
```

**For local development:**
```bash
git clone https://github.com/shaunakmirajgaonkar/Healthnexaa.git
cd Healthnexaa
pip install -e .
```

---

## Public API

```python
from medextract import (
    extract_folder,   # process all images in a folder
    analyze_image,    # process a single image
    classify_bp,      # get AHA category string
    validate_bp,      # get list of out-of-range warnings
    check_ollama,     # verify Ollama is running and model is pulled
)
```

---

## Option 1 — Use in Python Code

### Simplest usage — process a whole folder

```python
from medextract import extract_folder

rows = extract_folder("/path/to/bp-monitor-photos")

for row in rows:
    print(row["systolic"], "/", row["diastolic"], "  Pulse:", row["pulse"])
```

### Save results to a CSV

```python
import pandas as pd
from medextract import extract_folder

rows = extract_folder("/path/to/bp-monitor-photos")
df = pd.DataFrame(rows)
df.to_csv("results.csv", index=False)
print(df)
```

### Process a single image

```python
from medextract import analyze_image

result = analyze_image("/path/to/photo.jpg")

if result:
    print("Systolic :", result["systolic"])
    print("Diastolic:", result["diastolic"])
    print("Pulse    :", result["pulse"])
    print("Brand    :", result["brand"])
    print("Confidence:", result["confidence"], "/ 10")
```

### Validate and classify results

```python
from medextract import extract_folder, classify_bp, validate_bp

rows = extract_folder("/path/to/photos")

for row in rows:
    category = classify_bp(row["systolic"], row["diastolic"])
    warnings = validate_bp(row)

    print(f"{row['file_name']:30s}  {row['systolic']}/{row['diastolic']}  → {category}")
    if warnings:
        print("  WARNINGS:", warnings)
```

### 1000+ images — progress bar and resume

```python
from medextract import extract_folder

# tqdm progress bar shown automatically with live BP readout and ETA
rows = extract_folder("/path/to/1000-photos", workers=5)
```

Resume a crashed run without reprocessing completed images:

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
    workers=5,                   # parallel workers (default: 3)
    image_size=768,              # max px before encoding (default: 512)
    max_retries=5,               # retries per image on failure (default: 3)
    resume_csv="results.csv",    # path to existing CSV to resume from
)
```

**Input validation** — raises `ValueError` immediately if:
- `workers < 1`
- `image_size < 64`
- `max_retries < 1`

**Ollama check** — raises `RuntimeError` with a helpful message if:
- Ollama is not running (`ollama serve`)
- The model is not pulled (`ollama pull medgemma1.5:4b`)

---

## Option 2 — Use from the Terminal (no Python needed)

```bash
# Basic usage
python3 -m medextract.cli /path/to/photos

# Full options
python3 -m medextract.cli /path/to/photos \
    --output my_readings.csv \
    --workers 5 \
    --model medgemma1.5:4b \
    --image-size 768 \
    --max-retries 5 \
    --resume

# See all available options
python3 -m medextract.cli --help
```

---

## Output Fields

Each result dict (and each CSV row) contains these fields:

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
| `confidence` | int 1–10 | Model confidence (always clamped 1–10) |
| `bp_classification` | string | AHA category (Normal, Elevated, Stage 1/2, Crisis) |
| `extracted_at` | string | Timestamp of extraction (YYYY-MM-DD HH:MM:SS) |

---

## BP Classification Reference (AHA Standard)

| Classification | Systolic | Diastolic |
|---|---|---|
| Normal | < 120 | < 80 |
| Elevated | 120–129 | < 80 |
| Stage 1 Hypertension | 130–139 | 80–89 |
| Stage 2 Hypertension | ≥ 140 | ≥ 90 |
| HYPERTENSIVE CRISIS | > 180 | > 120 |

---

## Performance Reference (Apple Silicon)

| Workers | 100 images | 500 images | 1000 images |
|---|---|---|---|
| 3 (default) | ~19 min | ~97 min | ~3.2 hrs |
| 5 | ~12 min | ~58 min | ~1.9 hrs |
| 8 | ~7 min | ~36 min | ~1.2 hrs |

---

## Supported Image Formats

`.png`, `.jpg`, `.jpeg`, `.webp`

---

## Verified Test Results (2026-04-25)

### Live extraction on real devices

| Image | Device | BP | Pulse | Brand | Confidence | Category |
|---|---|---|---|---|---|---|
| Unknown-2.jpeg | Meditech BP-12 | 130/80 | 76 | MEDITECH | 10/10 | Stage 1 Hypertension |
| 20210805_..._F_3000_4000.jpg | Omron | 140/80 | 70 | Omron | 10/10 | Stage 2 Hypertension |
| Unknown-9.jpeg | Unknown (glare) | 0/0 | 0 | Unknown | 5/10 | Unknown (handled gracefully) |

### Automated test suite

| Test Group | Tests | Result |
|---|---|---|
| `classify_bp` — all AHA categories | 10 | PASS |
| `validate_bp` — boundary and inversion | 8 | PASS |
| Image load and base64 encode | 5 | PASS |
| `check_ollama` — running / not running / missing model | 3 | PASS |
| `analyze_image` — success, retry, markdown, confidence clamp | 5 | PASS |
| `extract_folder` — input validation | 4 | PASS |
| `extract_folder` — resume | 2 | PASS |
| **Total** | **37** | **ALL PASS** |

### Environment

| Item | Detail |
|---|---|
| Python | 3.12 |
| Ollama model | medgemma1.5:4b (3.3 GB) |
| Platform | macOS Darwin 25.4.0 |
| CI | GitHub Actions — Python 3.10, 3.11, 3.12 |
| Install method | `pip install -e .` (editable) |
