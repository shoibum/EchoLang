"""
Test module for Pipeline functionality.
"""

import os
import pytest
from pathlib import Path
from src.pipeline import Pipeline

@pytest.fixture
def pipeline():
    """Fixture to create a Pipeline instance."""
    return Pipeline()

def test_pipeline_initialization(pipeline):
    """Test that pipeline initializes properly."""
    assert pipeline.stt is not None
    assert pipeline.tts is not None
    assert pipeline.translator is not None

def test_supported_languages(pipeline):
    """Test supported languages."""
    languages = pipeline.supported_languages()
    assert "en" in languages
    assert "hi" in languages
    assert "kn" in languages

# The following tests require model downloads and are slow
@pytest.mark.slow
def test_text_translation(pipeline):
    """Test text translation."""
    translated = pipeline.translate_text("Hello, how are you?", "en", "hi")
    assert len(translated) > 0
    assert translated != "Hello, how are you?"  # Should be different from input