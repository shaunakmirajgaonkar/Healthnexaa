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


def check_ollama(model: str = "medgemma1.5:4b") -> None:
    """Raise a clear RuntimeError if Ollama is not running or the model is not pulled."""
    try:
        available = [m.model for m in ollama.list().models]
    except Exception:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            "Install from: https://ollama.com"
        )
    if not any(model in m for m in available):
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
    image_path: str | Path,
    model: str = "medgemma1.5:4b",
    image_size: int = 512,
    max_retries: int = 3,
) -> dict | None:
    """Extract BP readings from a single image file.

    Returns a dict of extracted fields, or None if the image could not be processed.
    Raises RuntimeError if Ollama is not running or the model is not available.
    """
    image_path = Path(image_path)
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
            # Clamp confidence to 1–10 regardless of what the model returns
            data["confidence"] = max(1, min(10, int(data.get("confidence", 5))))
            return data
        except Exception as e:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, max_retries, image_path.name, e)
            if attempt < max_retries:
                time.sleep(2)

    log.error("Skipping %s: all retries exhausted.", image_path.name)
    return None


def classify_bp(systolic: int, diastolic: int) -> str:
    """Return an AHA BP category string for the given systolic/diastolic values."""
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
    """Return a list of warning strings for any out-of-range BP/pulse values."""
    warnings = []
    s = row.get("systolic", 0)
    d = row.get("diastolic", 0)
    p = row.get("pulse", 0)
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
    """Process every supported image in `folder` and return a list of result dicts.

    Each dict contains the extracted BP fields plus:
      - bp_classification: AHA category string
      - extracted_at: timestamp string (YYYY-MM-DD HH:MM:SS)

    Raises RuntimeError if Ollama is not running or the model is not pulled.
    Raises FileNotFoundError if the folder does not exist.
    Supported formats: .png, .jpg, .jpeg, .webp
    """
    check_ollama(model)

    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = sorted(f for f in folder.iterdir() if f.suffix.lower() in EXTENSIONS)
    if not images:
        log.warning("No images found in %s", folder)
        return []

    log.info("Found %d image(s) in %s", len(images), folder)

    results: dict[int, dict] = {}

    def _process(args):
        idx, path = args
        log.info("Processing [%d]: %s", idx + 1, path.name)
        result = analyze_image(path, model=model, image_size=image_size, max_retries=max_retries)
        return idx, result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, (i, img)): i for i, img in enumerate(images)}
        for future in as_completed(futures):
            idx, result = future.result()
            if result:
                warnings = validate_bp(result)
                if warnings:
                    log.warning("[%s] Validation: %s", result.get("file_name"), "; ".join(warnings))
                result["bp_classification"] = classify_bp(
                    result.get("systolic", 0), result.get("diastolic", 0)
                )
                result["extracted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log.info(
                    "Done [%s]: BP %s/%s  Pulse %s  Confidence %s/10  [%s]",
                    result.get("file_name"),
                    result.get("systolic"),
                    result.get("diastolic"),
                    result.get("pulse"),
                    result.get("confidence"),
                    result.get("bp_classification"),
                )
                results[idx] = result
            else:
                log.error("Failed: %s", images[idx].name)

    return [results[i] for i in sorted(results)]
