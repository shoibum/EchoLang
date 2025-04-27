"""
EchoLang – central configuration
Last updated: 2025-04-27
"""

from pathlib import Path
import os

# ──────────────────────────── Paths ──────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent   # project root
DATA_DIR   = ROOT / "data"
AUDIO_DIR  = DATA_DIR / "audio"
MODELS_DIR = ROOT / "models"

for _dir in (DATA_DIR, AUDIO_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Languages ───────────────────────────
LANGUAGES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
}

# ─────────────────────────── Seamless-M4T ────────────────────────
# Default to the Medium checkpoint; override at runtime with
#   M4T_MODEL_ID=facebook/seamless-m4t-v2-large  …
M4T_MODEL_ID: str = os.getenv(
    "M4T_MODEL_ID",
    "facebook/hf-seamless-m4t-medium",
)

# ─────────────────────────── ASR models ──────────────────────────
WHISPER_MODEL_ID: str = "openai/whisper-medium"        # ~750 M params
KANNADA_W2V_ID:   str = "addy88/wav2vec2-kannada-stt"  # finetuned w2v2

# ─────────────────────────── MT models ───────────────────────────
# (src_lang, tgt_lang) ➜ Hugging Face checkpoint
TRANSLATION_MODELS: dict[tuple[str, str], str] = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "kn"): "ai4bharat/indictrans2-en-indic-1B",
    ("kn", "en"): "ai4bharat/indictrans2-indic-en-dist-200M",
}

# ─────────────────────────── TTS models ──────────────────────────
# per-language Coqui-TTS checkpoints (XTTS-v2 under the hood)
TTS_MODELS: dict[str, str] = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "hi": "tts_models/hi/gagan/tacotron2-DDC",
    "kn": "tts_models/multilingual/multi-dataset/your_tts",
}

# ─────────────────────────── Gradio UI ───────────────────────────
GRADIO_TITLE:       str = "EchoLang 🔊 — Multilingual Speech ⇆ Text"
GRADIO_DESCRIPTION: str = (
    "English · हिन्दी · ಕನ್ನಡ — record, translate or synthesise audio in one click!"
)
GRADIO_THEME:       str = "soft"

# ─────────────────────────── Exports ─────────────────────────────
__all__ = [
    "ROOT",
    "DATA_DIR",
    "AUDIO_DIR",
    "MODELS_DIR",
    "LANGUAGES",
    "M4T_MODEL_ID",
    "WHISPER_MODEL_ID",
    "KANNADA_W2V_ID",
    "TRANSLATION_MODELS",
    "TTS_MODELS",
    "GRADIO_TITLE",
    "GRADIO_DESCRIPTION",
    "GRADIO_THEME",
]