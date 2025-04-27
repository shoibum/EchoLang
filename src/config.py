"""
Global config – Seamless M4T-Large ASR + IndicTrans-2 + XTTS-v2.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
AUDIO_DIR    = DATA_DIR / "audio"
MODELS_DIR   = PROJECT_ROOT / "models"
for p in (DATA_DIR, AUDIO_DIR, MODELS_DIR): p.mkdir(parents=True, exist_ok=True)

LANGUAGES = {"en": "English", "hi": "Hindi", "kn": "Kannada"}

# ASR – we only keep the name for bookkeeping
ASR_MODEL_ID = "facebook/seamless-m4t-large"     # Dense, GPU-safe :contentReference[oaicite:1]{index=1}

# Translation (unchanged)
PAIR2MODEL = {
    ("en", "hi"): "ai4bharat/indictrans2-en-indic-1B",
    ("hi", "en"): "ai4bharat/indictrans2-indic-en-1B",
    ("en", "kn"): "ai4bharat/indictrans2-en-indic-1B",
    ("kn", "en"): "ai4bharat/indictrans2-indic-en-1B",
}

# TTS – XTTS-v2
TTS_MODEL_ID = "coqui/XTTS-v2"

GRADIO_THEME  = "gradio/dark"
GRADIO_TITLE  = "EchoLang · Seamless M4T + XTTS-v2"
GRADIO_DESCRIPTION = "GPU-accelerated speech ↔ text with Meta Seamless M4T-Large."