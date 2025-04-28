# src/stt/stt.py
import logging
from typing import Optional, Dict, Union
from pathlib import Path

# Updated import for the refactored FasterWhisper wrapper
from .faster_whisper_asr import FasterWhisperASRModel
from ..utils.model_utils import ModelManager # Potentially not used if config is sufficient
from ..utils.language import LanguageCode

# Import config to get model configurations
try:
    from .. import config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

class SpeechToText:
    """
    Speech-to-text interface for EchoLang.
    Manages multiple FasterWhisper model instances for different languages.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager # Keep if needed by other parts, else remove
        self.asr_models: Dict[str, FasterWhisperASRModel] = {}

        logger.info("Initializing multiple STT models...")

        # Load configurations for each required model
        stt_configs = config.FASTER_WHISPER_CONFIG

        required_keys = {
            LanguageCode.KANNADA: "kannada-small-ct2",
            LanguageCode.HINDI: "hindi-small-ct2",
            LanguageCode.ENGLISH: "base-small-ct2" # Use base for English and Default
        }

        for lang_code, config_key in required_keys.items():
            if config_key in stt_configs:
                logger.info(f"Loading STT model for language '{lang_code}' using config key '{config_key}'...")
                try:
                    model_config = stt_configs[config_key]
                    # Pass the specific config dictionary and key to the ASR model class
                    self.asr_models[lang_code] = FasterWhisperASRModel(
                        model_config=model_config,
                        model_key=config_key
                    )
                    # Optionally trigger loading immediately, or let it lazy load on first use
                    # self.asr_models[lang_code].load_model() # Uncomment to load at init
                except Exception as e:
                    logger.error(f"Failed to initialize STT model for lang '{lang_code}' with key '{config_key}': {e}", exc_info=True)
                    # Decide how to handle failure: maybe raise error or leave model out
            else:
                logger.error(f"Configuration key '{config_key}' needed for language '{lang_code}' not found in config.FASTER_WHISPER_CONFIG.")
                # Decide how to handle missing config

        if not self.asr_models.get(LanguageCode.ENGLISH):
             # Ensure we always have a fallback model (English/Base)
             raise RuntimeError("Failed to load the essential base/English STT model.")

        logger.info(f"Initialized STT models for languages: {list(self.asr_models.keys())}")


    def transcribe(self, audio_path: Union[str, Path],
                   src_lang: Optional[str] = None) -> Dict:
        """
        Transcribe audio file to text using the appropriate FasterWhisper model.

        Args:
            audio_path: Path to audio file
            src_lang: Source language code hint (e.g., 'kn', 'hi', 'en'). Determines model selection.

        Returns:
            Dictionary with transcription results {'text': str, 'language': str}
        """
        selected_model = None
        model_lang_key = src_lang

        # Select model based on language hint
        if src_lang and src_lang in self.asr_models:
            selected_model = self.asr_models[src_lang]
            logger.debug(f"Using STT model for hinted language: {src_lang}")
        else:
            # Fallback to English/Base model if hint is missing or not specifically handled
            selected_model = self.asr_models[LanguageCode.ENGLISH]
            model_lang_key = LanguageCode.ENGLISH # Log which model is actually used
            if src_lang:
                 logger.warning(f"No specific STT model loaded for hint '{src_lang}'. Falling back to '{model_lang_key}' model.")
            else:
                 logger.debug(f"No language hint provided. Using default '{model_lang_key}' model.")

        if selected_model is None:
             # This should not happen if the ENGLISH model loaded correctly
             logger.error("Could not select an appropriate STT model.")
             return {"text": "ERROR: No STT model available", "language": src_lang or "unknown"}

        # Perform transcription using the selected model instance
        return selected_model.transcribe(
            audio_path=audio_path,
            src_lang=src_lang # Pass original hint to faster-whisper within the selected model
        )