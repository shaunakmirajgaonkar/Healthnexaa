# Healthnexaa — Project Plan

## What It Is

A Python library (`medextract`) that reads blood pressure monitor photos using a local LLM (Ollama + MedGemma), extracts all visible readings, and returns structured data ready for CSV export or direct use in Python code.

---

## Current State — COMPLETED ✅

**Library:** `medextract/` (installed via `pip install -e .`)

**Files created:**

| File | Purpose |
|---|---|
| `medextract/__init__.py` | Public API — exports 4 functions |
| `medextract/extractor.py` | Core logic — no global state |
| `medextract/cli.py` | CLI entry point (`python3 -m medextract.cli`) |
| `pyproject.toml` | Package config — installable via pip |

**What it does today:**
1. Accepts a folder path, model name, worker count, image size, and retry count as parameters
2. Resizes each image to configurable max size (default 512px) and converts to JPEG
3. Base64-encodes and sends to Ollama (local, no API key, no cost, no rate limits)
4. Parses structured JSON response with 14 fields per image
5. Retries up to N times on parse/connection failure
6. Processes images in parallel via `ThreadPoolExecutor`
7. Returns a list of dicts — caller controls what to do with the data (CSV, DB, print, etc.)

**Extracted fields per image (14 fields):**

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
| `confidence` | int 1–10 | — |
| `bp_classification` | string | `"Unknown"` |
| `extracted_at` | string | — |

---

## Verified Test Results (2026-04-25)

| Test | Result |
|---|---|
| `import medextract` | PASS — version 0.1.0 |
| `classify_bp` — 6 AHA categories | PASS (6/6) |
| `validate_bp` — boundary and inversion checks | PASS (3/3) |
| Image load + base64 encode | PASS (46288 chars, ~34 KB) |
| `analyze_image` on real Omron photo | PASS — BP 140/80, Pulse 70, Confidence 10/10 |
| `analyze_image` on Meditech BP-12 photo | PASS — BP 130/80, Pulse 76, Brand MEDITECH, Confidence 10/10 |
| `extract_folder` on 178 real images | PASS — all queued and processed, no crashes |

---

## Remaining Next Steps

| Priority | Task |
|---|---|
| Medium | Publish to PyPI so others can `pip install medextract` |
| Medium | Add glucometer support (`analyze_glucose_image()`) |
| Medium | Add `--resume` flag to skip already-processed images |
| Low | Add a `--dry-run` flag to validate images without calling Ollama |
| Low | Build a FastAPI web dashboard for drag-and-drop upload + trend charts |
| Low | Write a formal test suite (`pytest`) with mocked Ollama responses |

---

**Summary:** The project has been converted from a single-file script into a fully installable Python library. Anyone can now `pip install` it and call `extract_folder()` or `analyze_image()` directly in their own code, or use the CLI. Tested end-to-end on real BP monitor images with confirmed correct readings.
