# src/tts/synthesizer.py
from typing import Optional, Dict, Union
from pathlib import Path
import logging

# Import both TTS model wrappers
from .xtts import XTTSModel
from .mms_tts import MMS_TTSModel # Import the new MMS model wrapper
from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode

logger = logging.getLogger(__name__)

class TextToSpeech:
    """
    Text-to-speech interface for EchoLang.
    Uses MMS-TTS for Kannada ('kn') and XTTS for other supported languages.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        # (Keep __init__ method exactly as in Response #28)
        self.model_manager = model_manager
        logger.info("Initializing TTS models (XTTS and MMS)...")
        self.xtts_model = None
        self.mms_model = None
        try:
            self.xtts_model = XTTSModel(self.model_manager)
            logger.info("XTTS model interface initialized.")
        except Exception as e:
             logger.error(f"Failed to initialize XTTS model interface: {e}", exc_info=True)
        try:
            self.mms_model = MMS_TTSModel()
            logger.info("MMS-TTS model interface initialized (for Kannada).")
        except Exception as e:
             logger.error(f"Failed to initialize MMS-TTS model interface: {e}", exc_info=True)
        if not self.xtts_model and not self.mms_model:
             logger.critical("Failed to initialize ANY TTS model interfaces. TTS will not function.")


    # --- MODIFIED Method Signature: Removed 'speed' ---
    def synthesize(self,
                 text: str,
                 lang: str = LanguageCode.ENGLISH,
                 speaker_audio: Optional[str] = None) -> Dict: # Removed speed=1.0 default
    # --- End Modification ---
        """
        Synthesize speech from text using the appropriate model based on language.
        """
        default_error = {"audio_path": None, "text": text, "language": lang, "error": "TTS synthesis failed."}

        if lang == LanguageCode.KANNADA:
            if self.mms_model:
                logger.info(f"Synthesizing Kannada text using MMS-TTS...")
                # MMS-TTS doesn't use speaker_audio or speed
                return self.mms_model.synthesize(text=text)
            else:
                logger.error("Kannada TTS requested, but MMS-TTS model is not available (failed to initialize).")
                default_error["error"] = "Kannada TTS model unavailable."
                return default_error
        else:
            # Use XTTS for other languages
            if self.xtts_model:
                logger.info(f"Synthesizing '{lang}' text using XTTS...")
                # --- MODIFIED Call: Removed 'speed' argument ---
                return self.xtts_model.synthesize(
                    text=text,
                    lang=lang,
                    speaker_audio=speaker_audio
                    # speed argument removed
                )
                # --- End Modification ---
            else:
                 logger.error(f"TTS for '{lang}' requested, but XTTS model is not available (failed to initialize).")
                 supported_xtts_langs = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
                 if lang not in supported_xtts_langs:
                     err_msg = f"Language '{lang}' is not supported by XTTS."
                     logger.warning(err_msg)
                     default_error["error"] = err_msg
                 else:
                     default_error["error"] = f"XTTS model unavailable for language '{lang}'."
                 return default_error