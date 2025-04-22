"""
Test module for Speech-to-Text functionality.
"""

import os
import pytest
from pathlib import Path
from src.stt.stt import SpeechToText

@pytest.fixture
def stt_model():
    """Fixture to create a SpeechToText instance with the 'tiny' model for faster testing."""
    return SpeechToText(model_size="tiny")

def test_stt_initialization(stt_model):
    """Test that STT model initializes properly."""
    assert stt_model.model_size == "tiny"
    assert stt_model.model is None  # Model should be lazy-loaded

def test_stt_model_loading(stt_model):
    """Test that model loads correctly."""
    stt_model.load_model()
    assert stt_model.model is not None

# This test requires an English audio file
# @pytest.mark.skipif(not os.path.exists("tests/data/test_en.wav"), reason="Test audio file not found")
# def test_english_transcription(stt_model):
#     """Test English transcription."""
#     result = stt_model.transcribe("tests/data/test_en.wav", language="en")
#     assert isinstance(result, dict)
#     assert "text" in result
#     assert len(result["text"]) > 0