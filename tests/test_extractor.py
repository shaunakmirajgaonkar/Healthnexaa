"""Tests for medextract.extractor — all Ollama calls are mocked."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from medextract import analyze_image, classify_bp, extract_folder, validate_bp
from medextract.extractor import load_image_b64, check_ollama


# ---------------------------------------------------------------------------
# classify_bp
# ---------------------------------------------------------------------------

class TestClassifyBP:
    def test_normal(self):
        assert classify_bp(115, 75) == "Normal"

    def test_elevated(self):
        assert classify_bp(125, 79) == "Elevated"

    def test_stage1(self):
        assert classify_bp(135, 85) == "Stage 1 Hypertension"

    def test_stage2_systolic(self):
        assert classify_bp(145, 70) == "Stage 2 Hypertension"

    def test_stage2_diastolic(self):
        assert classify_bp(120, 92) == "Stage 2 Hypertension"

    def test_crisis_systolic(self):
        assert classify_bp(185, 80) == "HYPERTENSIVE CRISIS"

    def test_crisis_diastolic(self):
        assert classify_bp(150, 125) == "HYPERTENSIVE CRISIS"

    def test_zero_values(self):
        assert classify_bp(0, 0) == "Unknown"

    def test_boundary_elevated_low(self):
        assert classify_bp(120, 79) == "Elevated"

    def test_boundary_stage1_low(self):
        assert classify_bp(130, 80) == "Stage 1 Hypertension"


# ---------------------------------------------------------------------------
# validate_bp
# ---------------------------------------------------------------------------

class TestValidateBP:
    def test_valid_reading_no_warnings(self):
        assert validate_bp({"systolic": 120, "diastolic": 80, "pulse": 72}) == []

    def test_systolic_too_high(self):
        warnings = validate_bp({"systolic": 350, "diastolic": 80, "pulse": 72})
        assert "systolic 350 out of range" in warnings

    def test_systolic_too_low(self):
        warnings = validate_bp({"systolic": 40, "diastolic": 30, "pulse": 72})
        assert "systolic 40 out of range" in warnings

    def test_diastolic_too_high(self):
        warnings = validate_bp({"systolic": 120, "diastolic": 210, "pulse": 72})
        assert "diastolic 210 out of range" in warnings

    def test_pulse_too_high(self):
        warnings = validate_bp({"systolic": 120, "diastolic": 80, "pulse": 280})
        assert "pulse 280 out of range" in warnings

    def test_systolic_less_than_diastolic(self):
        warnings = validate_bp({"systolic": 70, "diastolic": 90, "pulse": 72})
        assert "systolic must be greater than diastolic" in warnings

    def test_zero_values_no_warnings(self):
        # Zeros mean unreadable — should not trigger range warnings
        assert validate_bp({"systolic": 0, "diastolic": 0, "pulse": 0}) == []

    def test_multiple_warnings(self):
        warnings = validate_bp({"systolic": 400, "diastolic": 250, "pulse": 300})
        assert len(warnings) >= 3


# ---------------------------------------------------------------------------
# load_image_b64
# ---------------------------------------------------------------------------

class TestLoadImageB64:
    def test_valid_image(self, tmp_path):
        img_path = tmp_path / "test.jpg"
        Image.new("RGB", (200, 200), color=(100, 150, 200)).save(img_path)
        result = load_image_b64(img_path)
        assert result is not None
        assert len(result) > 100
        # Must be valid base64
        base64.b64decode(result)

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img_path = tmp_path / "test.png"
        Image.new("RGBA", (100, 100), color=(100, 150, 200, 128)).save(img_path)
        result = load_image_b64(img_path)
        assert result is not None

    def test_large_image_resized(self, tmp_path):
        img_path = tmp_path / "large.jpg"
        Image.new("RGB", (4000, 3000), color=(50, 50, 50)).save(img_path)
        result = load_image_b64(img_path, image_size=512)
        assert result is not None
        # Decode and check size stayed within 512
        decoded = base64.b64decode(result)
        from io import BytesIO
        img = Image.open(BytesIO(decoded))
        assert max(img.size) <= 512

    def test_invalid_file_returns_none(self, tmp_path):
        bad_path = tmp_path / "not_an_image.txt"
        bad_path.write_text("this is not an image")
        result = load_image_b64(bad_path)
        assert result is None

    def test_missing_file_returns_none(self, tmp_path):
        result = load_image_b64(tmp_path / "nonexistent.jpg")
        assert result is None


# ---------------------------------------------------------------------------
# check_ollama
# ---------------------------------------------------------------------------

class TestCheckOllama:
    def test_raises_if_ollama_not_running(self):
        with patch("ollama.list", side_effect=Exception("Connection refused")):
            with pytest.raises(RuntimeError, match="Ollama is not running"):
                check_ollama()

    def test_raises_if_model_not_pulled(self):
        mock_model = MagicMock()
        mock_model.model = "llama3:latest"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        with patch("ollama.list", return_value=mock_response):
            with pytest.raises(RuntimeError, match="not pulled"):
                check_ollama("medgemma1.5:4b")

    def test_passes_if_model_available(self):
        mock_model = MagicMock()
        mock_model.model = "medgemma1.5:4b"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        with patch("ollama.list", return_value=mock_response):
            check_ollama("medgemma1.5:4b")  # should not raise


# ---------------------------------------------------------------------------
# analyze_image
# ---------------------------------------------------------------------------

MOCK_RESPONSE = {
    "message": {
        "content": '{"systolic": 130, "diastolic": 80, "pulse": 72, "brand": "Omron", '
                   '"date": null, "time": null, "memory_slot": null, "ihb": false, '
                   '"afib": false, "error_code": null, "user": null, "battery_low": false, '
                   '"has_glare": false, "confidence": 9}'
    }
}


class TestAnalyzeImage:
    def test_returns_dict_on_success(self, tmp_path):
        img_path = tmp_path / "bp.jpg"
        Image.new("RGB", (200, 200), color=(200, 200, 200)).save(img_path)
        with patch("ollama.chat", return_value=MOCK_RESPONSE):
            result = analyze_image(img_path)
        assert result is not None
        assert result["systolic"] == 130
        assert result["diastolic"] == 80
        assert result["pulse"] == 72
        assert result["brand"] == "Omron"
        assert result["file_name"] == "bp.jpg"

    def test_confidence_clamped_to_1_10(self, tmp_path):
        img_path = tmp_path / "bp.jpg"
        Image.new("RGB", (200, 200)).save(img_path)
        high_conf = dict(MOCK_RESPONSE["message"])
        high_conf["content"] = MOCK_RESPONSE["message"]["content"].replace('"confidence": 9', '"confidence": 100')
        with patch("ollama.chat", return_value={"message": high_conf}):
            result = analyze_image(img_path)
        assert result["confidence"] == 10

    def test_returns_none_for_invalid_image(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_text("not an image")
        result = analyze_image(bad)
        assert result is None

    def test_returns_none_after_all_retries_fail(self, tmp_path):
        img_path = tmp_path / "bp.jpg"
        Image.new("RGB", (200, 200)).save(img_path)
        with patch("ollama.chat", side_effect=Exception("Model error")):
            with patch("time.sleep"):  # skip actual sleep in tests
                result = analyze_image(img_path, max_retries=2)
        assert result is None

    def test_strips_markdown_fences(self, tmp_path):
        img_path = tmp_path / "bp.jpg"
        Image.new("RGB", (200, 200)).save(img_path)
        fenced = {
            "message": {
                "content": "```json\n" + MOCK_RESPONSE["message"]["content"] + "\n```"
            }
        }
        with patch("ollama.chat", return_value=fenced):
            result = analyze_image(img_path)
        assert result is not None
        assert result["systolic"] == 130


# ---------------------------------------------------------------------------
# extract_folder — resume
# ---------------------------------------------------------------------------

class TestExtractFolderValidation:
    def test_invalid_workers_raises(self, tmp_path):
        with pytest.raises(ValueError, match="workers must be >= 1"):
            extract_folder(tmp_path, workers=0)

    def test_invalid_image_size_raises(self, tmp_path):
        with pytest.raises(ValueError, match="image_size must be >= 64"):
            extract_folder(tmp_path, image_size=10)

    def test_invalid_max_retries_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_retries must be >= 1"):
            extract_folder(tmp_path, max_retries=0)

    def test_missing_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            mock_model = MagicMock()
            mock_model.model = "medgemma1.5:4b"
            mock_response = MagicMock()
            mock_response.models = [mock_model]
            with patch("ollama.list", return_value=mock_response):
                extract_folder("/nonexistent/path/xyz")


class TestExtractFolderResume:
    def _make_images(self, folder: Path, count: int):
        for i in range(count):
            Image.new("RGB", (100, 100), color=(i * 10, 0, 0)).save(folder / f"img_{i:03d}.jpg")

    def test_skips_already_done_images(self, tmp_path):
        self._make_images(tmp_path, 3)

        # Simulate a CSV that already has img_000 and img_001
        import pandas as pd
        csv_path = tmp_path / "out.csv"
        pd.DataFrame([
            {"file_name": "img_000.jpg", "systolic": 120, "diastolic": 80},
            {"file_name": "img_001.jpg", "systolic": 130, "diastolic": 85},
        ]).to_csv(csv_path, index=False)

        mock_model = MagicMock()
        mock_model.model = "medgemma1.5:4b"
        mock_response = MagicMock()
        mock_response.models = [mock_model]

        with patch("ollama.list", return_value=mock_response):
            with patch("ollama.chat", return_value=MOCK_RESPONSE):
                rows = extract_folder(tmp_path, resume_csv=csv_path)

        # Should have 3 total: 2 from CSV + 1 new
        assert len(rows) == 3
        file_names = [r["file_name"] for r in rows]
        assert "img_000.jpg" in file_names
        assert "img_001.jpg" in file_names
        assert "img_002.jpg" in file_names

    def test_no_resume_csv_processes_all(self, tmp_path):
        self._make_images(tmp_path, 2)

        mock_model = MagicMock()
        mock_model.model = "medgemma1.5:4b"
        mock_response = MagicMock()
        mock_response.models = [mock_model]

        with patch("ollama.list", return_value=mock_response):
            with patch("ollama.chat", return_value=MOCK_RESPONSE):
                rows = extract_folder(tmp_path)

        assert len(rows) == 2
