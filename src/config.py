# src/config.py
"""
EchoLang – central configuration (Updated for Distilled IndicTrans2 Models on CPU)
"""

from pathlib import Path
import torch
from typing import Optional, Dict # Added Dict

# ──────────────────────────── Paths ──────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
AUDIO_DIR   = DATA_DIR / "audio"
MODELS_DIR = ROOT / "models" # Stores converted STT and downloaded XTTS models

for _dir in (DATA_DIR, AUDIO_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Languages ───────────────────────────
LANGUAGES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
}
# Specific codes required by IndicTrans2
INDIC_TRANS_LANG_CODES: Dict[str, str] = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "kn": "kan_Knda",
    # Add other IndicTrans2 supported codes here if needed
}

# ─────────────────────────── Device & Precision ──────────────────
# Force CPU to avoid MPS errors with translation/TTS models
APP_DEVICE = "cpu"
APP_TORCH_DTYPE = torch.float32 # Use float32 for CPU


# ────────────────────── FasterWhisper (ASR) ─────────────────────
# Keep the multi-model STT configuration (already set to CPU)
FASTER_WHISPER_CONFIG = {
    "kannada-small-ct2": { "model_path": str(ROOT / "models/kannada-small-ct2"), "device": "cpu", "compute_type": "int8" },
    "hindi-small-ct2": { "model_path": str(ROOT / "models/hindi-small-ct2"), "device": "cpu", "compute_type": "int8" },
    "base-small-ct2": { "model_path": str(ROOT / "models/base-small-ct2"), "device": "cpu", "compute_type": "int8" }
}

# ─────────────────── Translation Models (IndicTrans2 Local) ────
# Define IndicTrans2 DISTILLED model IDs
INDIC_TRANS_EN_INDIC_MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
INDIC_TRANS_INDIC_EN_MODEL_ID = "ai4bharat/indictrans2-indic-en-dist-200M"
INDIC_TRANS_INDIC_INDIC_MODEL_ID = "ai4bharat/indictrans2-indic-indic-dist-320M"


# ─────────────────────────── XTTS-v2 (TTS) ───────────────────────
# Keep XTTS config (XTTS wrapper already forces CPU)
XTTS_V2_CONFIG = { "default": { "id": "xtts_v2",
    "model_url": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
    "config_url": "https://huggingface.co/coqui/XTTS-v2/raw/main/config.json",
    "vocab_url": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    "local_dir": "xtts_v2", "model_filename": "model.pth",
    "config_filename": "config.json", "vocab_filename": "vocab.json",
}}
DEFAULT_XTTS_MODEL_KEY = "default"


# ─────────────────────────── Gradio UI ───────────────────────────
GRADIO_TITLE:       str = "EchoLang 🔊 (Distilled IndicTrans2 - CPU)" # Updated title
GRADIO_DESCRIPTION: str = (
    "Multilingual Speech ⇆ Text (English · हिन्दी · ಕನ್ನಡ)\n"
    "Powered by FasterWhisper (STT), Distilled IndicTrans2 (Translation on CPU), MMS-TTS (kn TTS on CPU), and XTTS-v2 (en/hi TTS on CPU)." # Updated description
)
GRADIO_THEME:       str = "soft"

# ─────────────────────────── Exports ─────────────────────────────
__all__ = [
    # Paths
    "ROOT", "DATA_DIR", "AUDIO_DIR", "MODELS_DIR",
    # Languages
    "LANGUAGES", "INDIC_TRANS_LANG_CODES",
    # Device
    "APP_DEVICE", "APP_TORCH_DTYPE", # Reflects CPU setting
    # Model Configs
    "FASTER_WHISPER_CONFIG",
    "INDIC_TRANS_EN_INDIC_MODEL_ID", # Distilled models
    "INDIC_TRANS_INDIC_EN_MODEL_ID",
    "INDIC_TRANS_INDIC_INDIC_MODEL_ID",
    "XTTS_V2_CONFIG", "DEFAULT_XTTS_MODEL_KEY",
    # Gradio
    "GRADIO_TITLE", "GRADIO_DESCRIPTION", "GRADIO_THEME",
]