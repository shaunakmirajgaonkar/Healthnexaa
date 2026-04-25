# Complete Guide: Medical Device Image Extraction
### BP Monitor & Glucometer — Approaches, Methods & Development Plan

---

## 1. The Problem Space

Medical devices like BP monitors and glucometers display critical health data on small LCD/LED screens. Patients read these manually and either forget, misrecord, or never share them with doctors. The goal is to **automatically extract those numbers from photos** and turn them into structured, analyzable health data.

This is harder than general OCR because:
- Displays use **7-segment LCD fonts** (unusual letterforms)
- **Glare and reflections** from glass/plastic covers
- **Angle distortion** — phones rarely held perfectly straight
- **Low contrast** — LCD segments on grey background
- **Multiple reading types** on screen at once (systolic, diastolic, pulse all simultaneously)
- **Device diversity** — hundreds of brands with different layouts

---

## 2. Medical Devices in Scope

### Blood Pressure Monitors

| Type | Display | Common Brands |
|---|---|---|
| Upper arm automatic | 7-segment LCD, 3 large numbers | Omron, Withings, A&D, Beurer |
| Wrist automatic | Smaller LCD | Omron, Andes |
| Manual (sphygmomanometer) | Analog dial | Welch Allyn |

**Fields on screen (full extraction schema — 14 fields):**

```json
{
  "systolic": int,
  "diastolic": int,
  "pulse": int,
  "brand": "string",
  "date": "YYYY-MM-DD",
  "time": "HH:MM",
  "memory_slot": "M1 / M2 / null",
  "ihb": bool,
  "afib": bool,
  "error_code": "E1–E5 / null",
  "user": "User 1 / User 2 / null",
  "battery_low": bool,
  "has_glare": bool,
  "confidence": "1–10"
}
```

### Glucometers (Blood Glucose Monitors)

| Type | Display | Common Brands |
|---|---|---|
| Standard glucometer | Small 7-segment LCD | Accu-Chek, OneTouch, FreeStyle, Contour |
| Continuous CGM reader | Color LCD with graph | FreeStyle Libre, Dexcom |
| Smart glucometer | Large color screen | LifeScan, Dario |

**Fields on screen (full extraction schema):**

```json
{
  "glucose_value": "number or null if HI/LO/error",
  "unit": "mg/dL or mmol/L",
  "is_high": bool,
  "is_low": bool,
  "meal_marker": "pre-meal / post-meal / none / unknown",
  "date": "YYYY-MM-DD or null",
  "time": "HH:MM or null",
  "average_readings": "number or null",
  "control_solution": bool,
  "battery_low": bool,
  "error_code": "string or null",
  "brand": "string or Unknown",
  "confidence": "1–10"
}
```

### Other Extractable Devices (Future Scope)
- Pulse oximeters (SpO2 + pulse)
- Digital thermometers
- Weighing scales (body weight)
- Peak flow meters (respiratory)
- Cholesterol meters

---

## 3. Technical Approaches — All Methods Explained

---

### Approach 1: Local LLM (Ollama + MedGemma) ⭐ Current Approach

**How it works:**
Send the image to a local vision model via Ollama with a prompt asking it to read the display and return structured JSON. No data leaves the machine.

**Current implementation uses `medgemma1.5:4b`** — a medical vision model optimized for reading clinical images including device displays. Free to run, no API key, no rate limits.

```python
from medextract import extract_folder

rows = extract_folder("/path/to/bp-photos", workers=5)
for row in rows:
    print(row["systolic"], "/", row["diastolic"], "—", row["bp_classification"])
```

**Prerequisites:**
```bash
ollama serve
ollama pull medgemma1.5:4b
pip install git+https://github.com/shaunakmirajgaonkar/Healthnexaa.git
```

**Pros:**
- No data leaves the machine — HIPAA-friendly
- No API key, no cost, no rate limits
- Works on any device, any angle, any lighting
- Extracts all 14 fields in one call
- Understands symbols (IHB, AFib, error codes)
- Confidence clamped to 1–10 for consistent output

**Cons:**
- Requires Ollama (3.3 GB model download, one-time)
- Slower than local OCR (~11 sec/image on Apple Silicon)
- Occasional hallucinations on very blurry or glare-heavy images

**Best for:** The current `medextract` library — local, private, free, production-ready

---

### Approach 2: Cloud LLM Vision API (Gemini / GPT-4o / Claude)

**How it works:**
Send the image to a multimodal LLM API with a structured prompt. Works the same as the local approach but images go to a third-party server.

```python
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

img = Image.open("bp_reading.jpg")
response = model.generate_content([prompt, img])
```

**Pros:**
- Works on almost any device, any angle, any lighting
- Zero training data needed
- Handles glare, rotation, partial occlusion
- Extracts ALL fields in one call

**Cons:**
- Images sent to third-party servers (privacy concern for PHI)
- Free tiers have rate limits (Gemini: 15 RPM, 1,500/day)
- Needs internet connection
- API cost at scale (Gemini Flash: ~$0.00035/image, GPT-4o: ~$0.01/image)

**Best for:** Rapid prototyping or use cases where local compute is unavailable

---

### Approach 3: Traditional OCR (Tesseract / Cloud OCR)

**How it works:**
Pre-process the image to isolate the digit region, then run an OCR engine to read the characters.

```python
import pytesseract
from PIL import Image
import cv2

img = cv2.imread('bp_reading.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
text = pytesseract.image_to_string(
    thresh,
    config='--psm 7 -c tessedit_char_whitelist=0123456789'
)
```

**Pros:**
- Free (Tesseract) or cheap (cloud OCR)
- Fast (sub-100ms locally)
- Data stays local with Tesseract
- No hallucinations — reads what's there

**Cons:**
- Fails on glare, poor angle, unusual fonts
- 7-segment LCD fonts confuse standard OCR
- Requires significant image pre-processing
- Each device layout needs custom region extraction
- Cannot interpret symbols (IHB, error codes)

**Best for:** High volume, controlled photography conditions, cost-sensitive deployments

---

### Approach 4: Custom Trained Deep Learning (CNN + Object Detection)

**How it works:**
Two-stage pipeline:
1. **Detection** — YOLO or similar finds the digit display region in the photo
2. **Recognition** — A classifier reads each digit from the detected region

```
Photo → YOLO (detect display) → Crop → Digit Segmenter → Classifier per digit → Value
```

**Training data needed:**
- 500–2000 labeled images minimum
- Labels: bounding boxes around display, digit values
- Can augment: brightness, rotation, glare simulation, blur

**Pros:**
- Extremely fast once trained (10–50ms/image)
- Runs fully offline, no API cost
- Highest possible accuracy on in-distribution devices
- HIPAA-friendly (no external data transfer)
- Can run on edge devices (Raspberry Pi, phone)

**Cons:**
- Significant upfront effort (data collection, labeling, training)
- Only works well on device types it was trained on
- Needs retraining for new devices
- Requires ML expertise

**Best for:** Production at scale, privacy-critical deployments, specific device types

---

### Approach 5: Hybrid Pipeline (Recommended for Future Scale)

**How it works:**
Combine approaches based on confidence:

```
Image
  → Pre-processing (denoise, deskew, enhance)
  → OCR attempt (fast, cheap)
    → if confidence > threshold → accept result
    → if confidence low → send to local LLM (accurate)
  → Post-processing (validate ranges, flag outliers)
  → Store result
```

**Routing logic:**
```python
def extract(image):
    result = ocr_extract(image)
    if result.confidence > 0.85 and result.in_valid_range():
        return result
    else:
        return llm_extract(image)  # fallback
```

This gives speed and cost efficiency for clean images, accuracy for hard ones.

**Best for:** Production systems at medium-to-large scale where throughput matters

---

### Approach 6: Specialized Medical / Device APIs

Companies building specifically for medical device data:

| Service | Focus |
|---|---|
| Infermedica API | Medical data extraction |
| Human API | Health data aggregation |
| Validic | Connected health device data |
| Dexcom API | CGM data (Dexcom devices only) |
| Withings API | Withings device data |
| Apple HealthKit | iPhone health data integration |
| Google Fit API | Android health data |

**Best for:** If the patient already uses a smart device that has an official API — skip image extraction entirely and pull data directly from the device's cloud.

---

### Approach 7: On-Device Mobile Processing

**How it works:**
Process the image directly on the user's phone before it ever leaves:

- **iOS:** Core ML + Vision framework
- **Android:** ML Kit (Google) + TensorFlow Lite
- **Cross-platform:** Flutter + TFLite or ONNX Runtime Mobile

```swift
// iOS example
let request = VNRecognizeTextRequest { request, error in
    let observations = request.results as? [VNRecognizedTextObservation]
    // parse BP values from recognized text
}
```

**Pros:**
- No data leaves the device
- Works offline
- Instant (no API latency)
- Privacy-preserving

**Cons:**
- Requires mobile app development
- Limited model size (phones have constraints)
- Harder to update model after deployment

**Best for:** Consumer-facing mobile apps where privacy is paramount

---

## 4. Image Pre-Processing Pipeline (Deep Dive)

Good pre-processing dramatically improves accuracy for all approaches.

```
Raw Photo
    │
    ▼
1. EXIF rotation correction     ← phones rotate images incorrectly
    │
    ▼
2. Resize / normalize           ← max 1024px, consistent input
    │
    ▼
3. Denoise                      ← remove camera noise
    │
    ▼
4. Display region detection     ← crop out irrelevant background
    │
    ▼
5. Perspective correction       ← deskew if image taken at angle
    │
    ▼
6. Contrast enhancement         ← make digits pop
    │
    ▼
7. Glare detection & masking    ← identify and handle reflections
    │
    ▼
8. Binarization                 ← convert to pure black/white (for OCR)
    │
    ▼
Ready for extraction
```

**Code example for key steps:**

```python
import cv2
import numpy as np
from PIL import Image, ImageOps

def preprocess(image_path):
    # EXIF rotation fix
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)

    # Resize
    img.thumbnail((1024, 1024), Image.LANCZOS)

    # Convert to OpenCV
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Denoise
    cv_img = cv2.fastNlMeansDenoisingColored(cv_img, None, 10, 10, 7, 21)

    # Enhance contrast via CLAHE
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    cv_img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    return cv_img
```

---

## 5. Post-Processing & Validation

After extraction, always validate the values against physiological limits.

```python
def validate_bp(systolic, diastolic, pulse):
    errors = []
    if not (60 <= systolic <= 300):
        errors.append(f"Systolic {systolic} out of physiological range")
    if not (30 <= diastolic <= 200):
        errors.append(f"Diastolic {diastolic} out of physiological range")
    if not (30 <= pulse <= 250):
        errors.append(f"Pulse {pulse} out of physiological range")
    if systolic <= diastolic:
        errors.append("Systolic must be greater than diastolic")
    return errors

def validate_glucose(value, unit):
    if unit == "mg/dL":
        if not (20 <= value <= 600):
            return ["Glucose value out of measurable range"]
    elif unit == "mmol/L":
        if not (1.1 <= value <= 33.3):
            return ["Glucose value out of measurable range"]
    return []
```

### AHA Blood Pressure Classification

```python
def classify_bp(systolic, diastolic):
    if systolic > 180 or diastolic > 120:
        return "HYPERTENSIVE CRISIS — seek emergency care"
    elif systolic >= 140 or diastolic >= 90:
        return "Stage 2 Hypertension"
    elif systolic >= 130 or diastolic >= 80:
        return "Stage 1 Hypertension"
    elif systolic >= 120 and diastolic < 80:
        return "Elevated"
    else:
        return "Normal"
```

### Blood Glucose Classification

```python
def classify_glucose(value, unit="mg/dL"):
    if unit == "mmol/L":
        value = value * 18.018  # convert to mg/dL
    if value < 70:    return "Hypoglycemia (Low) — take action"
    elif value < 100: return "Normal (Fasting)"
    elif value < 126: return "Prediabetes Range"
    else:             return "Diabetes Range"
```

---

## 6. Glucometer-Specific Considerations

Glucometers are harder than BP monitors because:

1. **Unit ambiguity** — mg/dL vs mmol/L look similar; wrong unit = catastrophically wrong interpretation (e.g., 7.0 mmol/L = 126 mg/dL, not 7 mg/dL)
2. **HI/LO indicators** — some values display as "HI" or "LO" text, not numbers
3. **Very small displays** — less pixel data to work with
4. **Meal markers** — fork/sun icons indicating pre/post meal

**Prompt template for glucometer:**

```python
glucose_prompt = """Read this glucometer display. Return ONLY JSON:
{
  "glucose_value": <number, or null if HI/LO/error>,
  "unit": "<mg/dL or mmol/L>",
  "is_high": <true if display shows HI>,
  "is_low": <true if display shows LO>,
  "meal_marker": "<pre-meal, post-meal, none, unknown>",
  "date": "<YYYY-MM-DD or null>",
  "time": "<HH:MM or null>",
  "error_code": "<string or null>",
  "brand": "<string or Unknown>",
  "confidence": <1-10>
}"""
```

---

## 7. Full System Architecture Options

### Option A — Python Library (Current — Production Ready ✅)
```
Local images → medextract library → list of dicts → CSV / DB / print
```
Good for: Any Python project, batch processing, research

### Option B — CLI Tool (Current — Production Ready ✅)
```
python3 -m medextract.cli /path/to/photos --output results.csv
```
Good for: Regular personal use, sharing with non-developers, automation scripts

### Option C — Local Web App (Planned)
```
Browser → FastAPI → Processing → SQLite → Chart Dashboard
```
Good for: Family use, single patient + doctor pair

### Option D — Multi-Patient Cloud App
```
Mobile App → API Gateway → Processing Service → PostgreSQL → Dashboard
                                    ↓
                          Notification Service (alerts for dangerous readings)
```
Good for: Clinic use, multiple patients

### Option E — Telegram / WhatsApp Bot
```
Patient sends photo to bot
        ↓
Bot extracts values → Bot replies with result + classification
        ↓
Stored in DB → Weekly summary sent automatically
```
Good for: Maximum ease of use, no app installation required

---

## 8. Technology Stack Recommendations

| Layer | Recommended | Alternative |
|---|---|---|
| Extraction | Ollama + MedGemma (local, current) | Gemini Flash (cloud, fast) |
| Backend | FastAPI (Python) | Flask, Django |
| Database | PostgreSQL | SQLite (small scale), MongoDB |
| Image processing | Pillow + OpenCV | scikit-image |
| Task queue | Celery + Redis | RQ (simple) |
| Frontend | React + Recharts | Streamlit (quick) |
| Auth | Supabase Auth | Firebase Auth |
| File storage | AWS S3 | Cloudinary |
| Deployment | Railway / Render | AWS / GCP |
| Monitoring | Sentry | Papertrail |

---

## 9. Privacy & Security Considerations

Medical image data is sensitive. Key concerns and mitigations:

| Risk | Mitigation |
|---|---|
| API key in source code | Move to `.env` file, never commit to git |
| Patient images sent to external servers | Use local Ollama model — no data leaves machine |
| CSV stored in plaintext | Encrypt at rest, restrict file permissions |
| No auth on web app | Add authentication before any web deployment |
| HIPAA compliance (US) | Local Ollama approach is HIPAA-friendly — no external transfer |

The `medextract` library uses Ollama exclusively — all processing is local, no data is sent to any external server.

---

## 10. API Cost Estimate (Cloud vs Local)

### Local Ollama (current approach):
| Volume | Cost |
|---|---|
| Any number of images | **$0** — runs on your hardware |

### Cloud LLM pricing (for comparison):

| Provider | Cost per image |
|---|---|
| Gemini 1.5 Flash | ~$0.00035 |
| GPT-4o (low detail) | ~$0.00255 |
| GPT-4o (high detail) | ~$0.01275 |

| Images | Gemini Flash | GPT-4o (low) | Local Ollama |
|---|---|---|---|
| 100 | ~$0.04 | ~$0.26 | **$0** |
| 1,000 | ~$0.35 | ~$2.55 | **$0** |
| 10,000 | ~$3.50 | ~$25.50 | **$0** |

---

## 11. Dataset & Training Resources (For Custom Model Route)

- **Roboflow Universe** — search "blood pressure monitor" for pre-labeled datasets
- **Kaggle** — medical device image datasets
- **PhysioNet** — clinical data (requires credentialing)
- **Synthetic data generation** — render fake BP monitor displays programmatically to bootstrap training
- **Data augmentation techniques:**
  - Glare simulation — white ellipse overlay at random positions
  - Rotation — ±30 degrees
  - Gaussian blur — simulate camera shake
  - Brightness variation — simulate poor lighting
  - Perspective transform — simulate off-angle photos

---

## 12. Development Roadmap

### Phase 1 — Initial Script ✅ COMPLETED (2026-04-20)
- Single-file script using Gemini 1.5 Flash
- 5-field extraction: systolic, diastolic, pulse, brand, has_glare
- Exponential backoff retry (up to 5 attempts)
- CSV export

### Phase 2 — Local Model + Hardening ✅ COMPLETED (2026-04-25)
- Switched to Ollama + MedGemma 1.5 4b — no API key, no cost, no rate limits
- 14-field extraction (added date, time, memory_slot, ihb, afib, error_code, user, battery_low, confidence)
- Parallel processing (`ThreadPoolExecutor`, 3 workers default)
- BP classification (Normal → HYPERTENSIVE CRISIS)
- Retry logic (configurable `max_retries`)
- Removed all hardcoded personal paths

### Phase 3 — Python Library ✅ COMPLETED (2026-04-25)
- Converted to installable `medextract` package
- Clean public API: `extract_folder()`, `analyze_image()`, `classify_bp()`, `validate_bp()`
- CLI entry point: `python3 -m medextract.cli`
- No global state — all config as function parameters with defaults
- `NullHandler` logging — caller controls logging
- Verified on real Omron and Meditech BP-12 images

**Verified results:**

| Image | BP | Pulse | Brand | Confidence | Category |
|---|---|---|---|---|---|
| Unknown-2.jpeg (Meditech BP-12) | 130/80 | 76 | MEDITECH | 10/10 | Stage 1 Hypertension |
| 20210805_..._F_3000_4000.jpg (Omron) | 140/80 | 70 | Omron | 10/10 | Stage 2 Hypertension |

### Phase 4 — Production Ready ✅ COMPLETED (2026-04-25)
- **37 pytest tests** — all passing, Ollama fully mocked (no model required to run tests)
- `check_ollama()` — friendly `RuntimeError` if Ollama not running or model not pulled
- **Input validation** — `ValueError` with clear message for invalid `workers`, `image_size`, `max_retries`
- **Confidence clamping** — always 1–10 regardless of model output (`max(1, min(10, value))`)
- **Pinned dependency versions** — `ollama>=0.6.1`, `pillow>=10.2.0`, `pandas>=2.1.1`, `tqdm>=4.66.1`
- **tqdm progress bar** — live BP readout and ETA during batch processing
- **Resume support** — `resume_csv` parameter skips images already in an existing CSV
- **`--resume` CLI flag** — resumes from existing `--output` CSV
- **GitHub Actions CI** — auto-runs tests on every push across Python 3.10, 3.11, 3.12
- **Version consistent** — `pyproject.toml` and `__init__.py` both report `0.3.0`
- Old scripts moved to `examples/` folder

### Phase 5 — FastAPI Web Dashboard (Planned)
- FastAPI backend with SQLite storage
- Drag-and-drop image upload in browser
- Trend charts (BP over time)
- Color-coded classification
- Export to CSV from UI

### Phase 6 — Glucometer Support (Planned)
- `analyze_glucose_image()` function
- New prompt template for glucometer displays
- Unit detection (mg/dL vs mmol/L)
- HI/LO indicator handling

### Phase 7 — Hybrid OCR Pipeline (Optional Future)
- Add Tesseract as fast-path for clean, well-lit images
- Call Ollama only when OCR confidence is low
- Reduces processing time by ~70% on good-quality images

### Phase 8 — Cloud & Multi-Patient (If Going Commercial)
- User accounts and authentication
- Mobile-friendly upload flow
- Doctor sharing link (read-only view)
- Push alerts for dangerous readings (Twilio SMS / email)
- Weekly PDF report generation

---

## 13. Summary — Best Approach

| Situation | Best Approach |
|---|---|
| Personal use, full privacy | Local Ollama + `medextract` library (current) |
| Family / small clinic | FastAPI web app + SQLite |
| Cloud-hosted, no local GPU | Gemini 1.5 Flash API |
| High volume (10k+ images/day) | Hybrid: Tesseract + LLM fallback |
| Mobile consumer app | On-device ML (Core ML / ML Kit) |
| Enterprise / HIPAA | Custom trained model + on-premise deployment |

**Current state:** The `medextract` library (Phase 4) is production-ready. It handles personal and small-clinic use cases fully — local, private, free, with no rate limits, 37 passing tests, and CI on GitHub.

---

*Document created: April 2026 | Project: Healthnexaa | Version: 0.4.0*
