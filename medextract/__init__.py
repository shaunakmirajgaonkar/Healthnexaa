"""medextract — extract readings from medical device images using a local LLM (Ollama)."""

from .extractor import analyze_image, check_ollama, classify_bp, extract_folder, validate_bp

__version__ = "0.1.0"
__all__ = ["extract_folder", "analyze_image", "classify_bp", "validate_bp", "check_ollama"]
