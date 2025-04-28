# src/config.py
"""
EchoLang – central configuration (Updated for LibreTranslate API)
"""

from pathlib import Path
import torch

# ──────────────────────────── Paths ──────────────────────────────
# (Keep this section as is)
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
AUDIO_DIR   = DATA_DIR / "audio"
MODELS_DIR = ROOT / "models"

for _dir in (DATA_DIR, AUDIO_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Languages ───────────────────────────
# (Keep this section as is)
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
}

# ─────────────────────────── Device & Precision ──────────────────
# (Keep this section as is)
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
# (Keep this section with the multiple models as configured before)
FASTER_WHISPER_CONFIG = {
    "kannada-small-ct2": { # Kannada specific
        "model_path": str(ROOT / "models/kannada-small-ct2"),
        "device": "cpu",
        "compute_type": "int8"
    },
    "hindi-small-ct2": { # Hindi specific
        "model_path": str(ROOT / "models/hindi-small-ct2"),
        "device": "cpu",
        "compute_type": "int8"
    },
    "base-small-ct2": { # Base multilingual (for English & Default)
        "model_path": str(ROOT / "models/base-small-ct2"),
        "device": "cpu",
        "compute_type": "int8"
    }
}

# ─────────────────── Translation Models (LibreTranslate API) ────
# --- REMOVED NLLB/MARIAN/MYMEMORY CONFIGS ---
LIBRETRANSLATE_URL = "https://libretranslate.com/" # Public instance


# ─────────────────────────── XTTS-v2 (TTS) ───────────────────────
# (Keep this section as is, referring to DEFAULT_XTTS_MODEL_KEY)
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
GRADIO_TITLE:       str = "EchoLang 🔊 (LibreTranslate API)" # Updated title
GRADIO_DESCRIPTION: str = (
    "Multilingual Speech ⇆ Text (English · हिन्दी · ಕನ್ನಡ)\n"
    "Powered by FasterWhisper (STT), LibreTranslate API (Translation), MMS-TTS (kn TTS), and XTTS-v2 (en/hi TTS)." # Updated description
)
GRADIO_THEME:       str = "soft"

# ─────────────────────────── Exports ─────────────────────────────
__all__ = [
    # Paths
    "ROOT", "DATA_DIR", "AUDIO_DIR", "MODELS_DIR",
    # Languages
    "LANGUAGES",
    # Device
    "APP_DEVICE", "APP_TORCH_DTYPE",
    # Model Configs
    "FASTER_WHISPER_CONFIG",
    "LIBRETRANSLATE_URL", # Added API URL
    "XTTS_V2_CONFIG", "DEFAULT_XTTS_MODEL_KEY",
    # Gradio
    "GRADIO_TITLE", "GRADIO_DESCRIPTION", "GRADIO_THEME",
]