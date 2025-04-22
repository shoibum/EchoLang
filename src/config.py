"""
Configuration settings for the multilingual STT/TTS project.
"""

import os
from pathlib import Path

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
AUDIO_DIR = DATA_DIR / "audio"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Supported languages
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada"
}

# Whisper model configurations
WHISPER_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large

# TTS configurations
TTS_MODELS = {
    "en": "tts_models/en/ljspeech/tacotron2-DDC",
    "hi": "tts_models/hi/gagan/tacotron2-DDC",
    "kn": "tts_models/multilingual/multi-dataset/your_tts"  # YourTTS can handle Kannada
}

# Translation model configurations
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-{src}-{tgt}"

# Web interface configurations
GRADIO_THEME = "default"
GRADIO_TITLE = "Multilingual Speech-to-Text and Text-to-Speech"
GRADIO_DESCRIPTION = "Convert speech to text and text to speech in English, Hindi, and Kannada with translation capabilities."