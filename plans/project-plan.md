# Healthnexaa — Project Plan

**Current version:** 0.3.0 | **Status:** Production Ready ✅

---

## What It Is

A fully installable Python library (`medextract`) that extracts blood pressure readings from medical device photos using a local AI model (Ollama + MedGemma). No API key, no internet connection, no cost. Works as a Python import or CLI command.

---

## Current State — All Phases Complete

**Library:** `medextract/` (installed via `pip install -e .` or from GitHub)

| File | Purpose |
|---|---|
| `medextract/__init__.py` | Public API — 5 exported functions |
| `medextract/extractor.py` | Core logic — no global state, input validation, resume, progress bar |
| `medextract/cli.py` | CLI entry point with `--resume` flag |
| `pyproject.toml` | Package config — pinned deps, dev group, CLI entry |
| `.github/workflows/tests.yml` | CI — auto-runs 37 tests on every push across Python 3.10/3.11/3.12 |
| `tests/test_extractor.py` | 37 tests — all functions fully covered, Ollama mocked |

**What it does:**
1. Validates parameters before processing (`ValueError` with clear message)
2. Checks Ollama is running and model is pulled (`RuntimeError` with fix instructions)
3. Resizes each image to configurable max size, converts to JPEG
4. Base64-encodes and sends to Ollama (local — no data leaves the machine)
5. Parses structured JSON response with 14 fields per image
6. Clamps confidence to 1–10 regardless of model output
7. Retries on failure with configurable max retries
8. Processes images in parallel via `ThreadPoolExecutor`
9. Shows tqdm progress bar with live BP readout and ETA
10. Supports resume — skips images already in an existing CSV
11. Returns list of dicts — caller controls output (CSV, DB, print, etc.)

---

## Extracted Fields (14 fields per image)

| Field | Type | Fallback |
|---|---|---|
| `file_name` | string | — |
| `systolic` | int | `0` |
| `diastolic` | int | `0` |
| `pulse` | int | `0` |
| `brand` | string | `"Unknown"` |
| `date` | string | `null` |
| `time` | string | `null` |
| `memory_slot` | string | `null` |
| `ihb` | bool | `false` |
| `afib` | bool | `false` |
| `error_code` | string | `null` |
| `user` | string | `null` |
| `battery_low` | bool | `false` |
| `has_glare` | bool | `false` |
| `confidence` | int 1–10 | clamped |
| `bp_classification` | string | `"Unknown"` |
| `extracted_at` | string | — |

---

## Phase History

### Phase 1 — Initial Script ✅
- Single-file Gemini API script extracting 5 fields
- Manual config, no retry, no logging

### Phase 2 — Local Model + Hardening ✅
- Switched to Ollama + MedGemma (no API key, no cost, no rate limits)
- 14-field extraction, parallel processing, retry logic, BP classification, validation

### Phase 3 — Python Library ✅
- Converted to installable `medextract` package
- Clean public API, no global state, NullHandler logging, CLI entry point
- Verified on real Omron and Meditech BP-12 images

### Phase 4 — Production Ready ✅ (2026-04-25)
- 37 pytest tests — all passing, Ollama fully mocked
- `check_ollama()` — friendly error if Ollama not running or model not pulled
- Confidence clamped to 1–10 in all cases
- Pinned dependency versions
- Old scripts moved to `examples/`
- GitHub Actions CI across Python 3.10, 3.11, 3.12
- Input validation with clear `ValueError` messages
- tqdm progress bar with live readout
- Resume support (`resume_csv`) for large batches
- Version consistent across `pyproject.toml` and `__init__.py`

---

## Verified Test Results (2026-04-25)

| Test | Result |
|---|---|
| 37 pytest tests | ALL PASS |
| Live extraction — Meditech BP-12 | BP 130/80, Pulse 76, Confidence 10/10 |
| Live extraction — Omron | BP 140/80, Pulse 70, Confidence 10/10 |
| 178-image folder batch | All processed, no crashes |
| CI (GitHub Actions) | Passing on Python 3.10, 3.11, 3.12 |

---

## Phase 5 — PyPI Publish ✅ (2026-04-25)

- Published to PyPI — `pip install medextract` works worldwide
- PyPI page: https://pypi.org/project/medextract/0.3.0/
- Fixed `pyproject.toml` license field to SPDX string format
- README and usage guide updated to show PyPI as primary install method

---

## Remaining Next Steps

| Priority | Task |
|---|---|
| Medium | Add glucometer support (`analyze_glucose_image()`) |
| Low | Add `--dry-run` flag to validate images without calling Ollama |
| Low | Build FastAPI web dashboard for drag-and-drop upload + trend charts |
| Low | Add Docker config for server deployment |
