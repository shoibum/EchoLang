# src/config.py
"""
EchoLang – central configuration (Updated for FasterWhisper, MarianMT/NLLB, XTTS)
"""

from pathlib import Path
import torch

# ──────────────────────────── Paths ──────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
AUDIO_DIR   = DATA_DIR / "audio"
MODELS_DIR = ROOT / "models" # Only XTTS models stored locally

for _dir in (DATA_DIR, AUDIO_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Languages ───────────────────────────
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
}
# NLLB specific codes (needed for NLLB fallback)
NLLB_LANG_CODES = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
}
# MarianMT generally uses simple 2-letter codes directly in the model name

# ─────────────────────────── Device & Precision ──────────────────
# Detect hardware
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    APP_DEVICE = "mps"
    APP_TORCH_DTYPE = torch.float16
elif torch.cuda.is_available():
    APP_DEVICE = "cuda"
    APP_TORCH_DTYPE = torch.float16
else:
    APP_DEVICE = "cpu"
    APP_TORCH_DTYPE = torch.float32

# ────────────────────── FasterWhisper (ASR) ─────────────────────
# Force CPU for faster-whisper due to MPS incompatibility observed
FASTER_WHISPER_CONFIG = {
    "small": {
        "model_size_or_path": "small",
        "device": "cpu",
        "compute_type": "int8" # Use int8 for optimized CPU inference
    }
}
DEFAULT_FASTER_WHISPER_MODEL_KEY = "small"

# ─────────────────── Translation Models ─────────────────────

# --- MarianMT Models (Helsinki-NLP) ---
# Specific models for supported language pairs
MARIAN_CONFIG = {
    "en-hi": {
        "hf_id": "Helsinki-NLP/opus-mt-en-hi",
        "model_type": "marian", # Add type for clarity
    },
    "hi-en": {
        "hf_id": "Helsinki-NLP/opus-mt-hi-en",
        "model_type": "marian",
    },
    # No dedicated en-kn or kn-en found, will use NLLB fallback
}

# --- NLLB Models (Facebook/Meta) ---
# Fallback for pairs not covered by MarianMT (e.g., Kannada)
NLLB_CONFIG = {
    "nllb-distilled-600M": {
        "hf_id": "facebook/nllb-200-distilled-600M",
        "model_type": "nllb", # Add type for clarity
        # Requires specific lang codes (see NLLB_LANG_CODES)
    },
}
# Define default NLLB model key to use as fallback
DEFAULT_NLLB_MODEL_KEY = "nllb-distilled-600M"


# ─────────────────────────── XTTS-v2 (TTS) ───────────────────────
# XTTS will run on CPU due to MPS incompatibility observed
XTTS_V2_CONFIG = {
    "default": {
        "id": "xtts_v2",
        "model_url": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
        "config_url": "https://huggingface.co/coqui/XTTS-v2/raw/main/config.json",
        "vocab_url": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
        "local_dir": "xtts_v2",
        "model_filename": "model.pth",
        "config_filename": "config.json",
        "vocab_filename": "vocab.json",
    }
}
DEFAULT_XTTS_MODEL_KEY = "default"

# ─────────────────────────── Gradio UI ───────────────────────────
GRADIO_TITLE:       str = "EchoLang 🔊 (FasterWhisper+Marian/NLLB+XTTS)" # Keep title
GRADIO_DESCRIPTION: str = (
    "Multilingual Speech ⇆ Text (English · हिन्दी · ಕನ್ನಡ)\n"
    "Powered by FasterWhisper, MarianMT (en↔hi), NLLB (en↔kn), and XTTS-v2." # Clarify model usage
)
GRADIO_THEME:       str = "soft"

# ─────────────────────────── Exports ─────────────────────────────
__all__ = [
    # Paths
    "ROOT", "DATA_DIR", "AUDIO_DIR", "MODELS_DIR",
    # Languages
    "LANGUAGES", "NLLB_LANG_CODES",
    # Device
    "APP_DEVICE", "APP_TORCH_DTYPE",
    # Model Configs
    "FASTER_WHISPER_CONFIG", "DEFAULT_FASTER_WHISPER_MODEL_KEY",
    "MARIAN_CONFIG", # Keep Marian
    "NLLB_CONFIG", "DEFAULT_NLLB_MODEL_KEY", # Keep NLLB
    "XTTS_V2_CONFIG", "DEFAULT_XTTS_MODEL_KEY",
    # Gradio
    "GRADIO_TITLE", "GRADIO_DESCRIPTION", "GRADIO_THEME",
]