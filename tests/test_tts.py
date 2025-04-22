"""
Test module for Text-to-Speech functionality.
"""

import os
import pytest
from pathlib import Path
from src.tts.tts import TextToSpeech

@pytest.fixture
def tts_model():
    """Fixture to create a TextToSpeech instance."""
    return TextToSpeech()

def test_tts_initialization(tts_model):
    """Test that TTS model initializes properly."""
    assert tts_model.synthesizers == {}

# This test takes time to run as it downloads models
@pytest.mark.slow
def test_tts_model_loading(tts_model):
    """Test that model loads correctly."""
    tts_model.load_model("en")
    assert "en" in tts_model.synthesizers

# This test actually synthesizes speech, so it's marked as slow
@pytest.mark.slow
def test_english_synthesis(tts_model):
    """Test English speech synthesis."""
    output_path = "tests/data/test_output_en.wav"
    result = tts_model.synthesize("Hello, this is a test.", "en", output_path)
    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
    
    # Clean up
    if os.path.exists(output_path):
        os.remove(output_path)