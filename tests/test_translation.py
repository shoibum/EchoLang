"""
Test module for Translation functionality.
"""

import pytest
from src.translation.translator import Translator

@pytest.fixture
def translator():
    """Fixture to create a Translator instance."""
    return Translator()

def test_translator_initialization(translator):
    """Test that translator initializes properly."""
    assert translator.translation_pipelines == {}
    assert len(translator.language_pairs) == 6  # 3 languages can have 6 pairs (3*2)

def test_model_name_generation(translator):
    """Test model name generation for language pairs."""
    model_name = translator.get_model_name("en", "hi")
    assert model_name == "Helsinki-NLP/opus-mt-en-hi"
    
    # Test special case for Hindi-Kannada
    model_name = translator.get_model_name("hi", "kn")
    assert model_name is None

# This test loads models and is slow
@pytest.mark.slow
def test_english_to_hindi_translation(translator):
    """Test English to Hindi translation."""
    translator.load_model("en", "hi")
    assert "en-hi" in translator.translation_pipelines
    
    result = translator.translate("Hello, how are you?", "en", "hi")
    assert len(result) > 0
    assert result != "Hello, how are you?"  # Should be different from input