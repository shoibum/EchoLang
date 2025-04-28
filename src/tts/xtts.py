# src/tts/xtts.py
import torch
import os
import tempfile
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import re
import logging
import time

# Use relative imports
from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode
try:
    from .. import config as main_config # Renamed to avoid clash
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config as main_config

logger = logging.getLogger(__name__)

try:
    from TTS.api import TTS
    logger.debug("Successfully imported TTS.api.TTS.")
except ImportError as e:
    logger.error("Failed to import TTS.api. Is TTS installed correctly? (pip install TTS)", exc_info=True)
    raise ImportError("Could not import TTS.api.") from e

class XTTSModel:
    """
    Wrapper for XTTS-v2 model using the high-level TTS.api interface.
    Handles model loading via API and synthesis using tts_to_file.
    FORCES CPU execution due to observed MPS incompatibilities.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        # (Keep __init__ method exactly as in Response #9)
        self.model_manager = model_manager or ModelManager()
        self.device = "cpu" # Override config.APP_DEVICE for XTTS
        self.tts_api: Optional[TTS] = None
        self.model_key = main_config.DEFAULT_XTTS_MODEL_KEY
        self.xtts_config = main_config.XTTS_V2_CONFIG.get(self.model_key)
        if not self.xtts_config:
            raise ValueError(f"Configuration for XTTS model key '{self.model_key}' not found in config.py.")
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info(f"Initializing XTTSModel wrapper for API model: '{self.model_name}' (FORCING CPU execution)")


    def load_model(self):
        # (Keep load_model method exactly as in Response #9)
        if self.tts_api is not None:
            logger.debug("TTS API instance already loaded.")
            return
        logger.info(f"Loading TTS API model: {self.model_name}")
        logger.info(f"Forcing device: {self.device} (Ignoring config.APP_DEVICE for XTTS)")
        try:
            start_load_time = time.time()
            self.tts_api = TTS(model_name=self.model_name, progress_bar=True, gpu=False)
            self.tts_api.to(self.device)
            end_load_time = time.time()
            logger.info(f"Successfully loaded TTS API model '{self.model_name}' onto CPU in {end_load_time - start_load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Error loading TTS API model '{self.model_name}': {e}", exc_info=True)
            self.tts_api = None
            raise RuntimeError(f"Failed to load TTS API model '{self.model_name}': {e}") from e

    def _get_reference_audio_path(self, lang_code: str) -> Optional[Path]:
        # (Keep _get_reference_audio_path method exactly as in Response #9)
        local_dir_name = self.xtts_config.get("local_dir", "xtts_v2")
        speaker_dir = self.model_manager.get_model_path(local_dir_name) / "speakers"
        ref_audio_path = speaker_dir / f"{lang_code}_reference.wav"
        return ref_audio_path

    def _prepare_reference_audio(self, lang_code: str) -> Optional[str]:
        # (Keep _prepare_reference_audio method exactly as in Response #9)
        ref_audio_path = self._get_reference_audio_path(lang_code)
        logger.debug(f"Looking for default speaker reference audio at: {ref_audio_path}")
        if ref_audio_path.exists():
            logger.info(f"Using default reference audio: {ref_audio_path}")
            return str(ref_audio_path)
        else:
            logger.error(f"Default reference speaker audio not found for language '{lang_code}'. Expected at: {ref_audio_path}. Cannot use default speaker.")
            return None

    # --- MODIFIED Method Signature: Removed 'speed' ---
    def synthesize(self,
                   text: str,
                   lang: str = LanguageCode.ENGLISH,
                   speaker_audio: Optional[str] = None) -> Dict: # Removed speed=1.0 default
    # --- End Modification ---
        """Synthesize speech from text using the TTS.api interface on CPU."""
        if self.tts_api is None:
            logger.info("TTS API instance not loaded. Calling load_model()...")
            try:
                 self.load_model()
            except Exception as load_err:
                 logger.error(f"Failed to load TTS API model during synthesize call: {load_err}", exc_info=True)
                 return {"audio_path": None, "text": text, "language": lang, "error": f"Model load failed: {type(load_err).__name__}"}
            if self.tts_api is None:
                 error_msg = "TTS API model could not be loaded. Cannot synthesize."
                 logger.error(error_msg)
                 return {"audio_path": None, "text": text, "language": lang, "error": error_msg}

        cleaned_text = self._clean_text(text)
        if not cleaned_text.strip():
            logger.warning("Input text is empty or only whitespace after cleaning.")
            return {"audio_path": None, "text": cleaned_text, "language": lang, "error": "Input text is empty"}

        speaker_wav_path = speaker_audio
        if speaker_wav_path:
             logger.info(f"Using provided speaker reference: {speaker_wav_path}")
             if not Path(speaker_wav_path).exists():
                  logger.error(f"Provided speaker audio file not found: {speaker_wav_path}")
                  return {"audio_path": None, "text": cleaned_text, "language": lang, "error": "Provided speaker audio not found"}
        else:
             logger.info(f"No speaker reference provided. Attempting default for language: {lang}")
             speaker_wav_path = self._prepare_reference_audio(lang)
             if speaker_wav_path is None:
                  # Check if the language *requires* a speaker (XTTS does)
                  supported_langs_no_default = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi'] # XTTS supported langs
                  if lang in supported_langs_no_default:
                     logger.error(f"Cannot synthesize: No speaker reference provided and default speaker for '{lang}' is missing. XTTS requires speaker input.")
                     return {"audio_path": None, "text": cleaned_text, "language": lang, "error": f"Speaker reference required for '{lang}' (default missing)"}
                  # else: # If we add models that don't need speakers, handle here
                  #    pass

        xtts_lang = lang.lower()
        logger.debug(f"Using language code for TTS API: '{xtts_lang}'")

        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                 output_path = tmp_file.name
            logger.debug(f"Created temporary output file: {output_path}")

            start_time = time.time()
            # --- MODIFIED Log: Removed 'speed' ---
            logger.info(f"Starting XTTS synthesis via API (CPU) for text: '{cleaned_text[:50]}...', Lang: {xtts_lang}")
            # --- End Modification ---

            # --- MODIFIED Call: Removed 'speed' argument ---
            self.tts_api.tts_to_file(
                text=cleaned_text,
                speaker_wav=speaker_wav_path,
                language=xtts_lang,
                file_path=output_path,
                # speed=speed, # Argument removed
                split_sentences=True
            )
            # --- End Modification ---

            end_time = time.time()
            logger.info(f"XTTS synthesis via API (CPU) successful in {end_time - start_time:.2f}s. Output: {output_path}")

            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                 logger.error("TTS synthesis seemed to succeed but output file is missing or empty.")
                 if Path(output_path).exists(): Path(output_path).unlink(missing_ok=True)
                 return {"audio_path": None, "text": cleaned_text, "language": lang, "error": "Synthesis failed (empty output file)"}

            return {
                "audio_path": output_path,
                "text": cleaned_text,
                "language": lang,
                "error": None
            }

        except FileNotFoundError as e:
             logger.error(f"File not found during synthesis: {e}", exc_info=True)
             if output_path and Path(output_path).exists(): Path(output_path).unlink(missing_ok=True)
             return {"audio_path": None, "text": cleaned_text, "language": lang, "error": f"File not found: {e.filename}"}
        except Exception as e:
            # Check for specific XTTS language support error before general failure
            if f"Language {lang} is not supported" in str(e):
                 logger.error(f"XTTS API reported language '{lang}' not supported: {e}", exc_info=True)
                 if output_path and Path(output_path).exists(): Path(output_path).unlink(missing_ok=True)
                 return {"audio_path": None, "text": cleaned_text, "language": lang, "error": f"XTTS does not support language '{lang}'"}

            logger.error(f"Error during TTS API synthesis process: {e}", exc_info=True)
            if output_path and Path(output_path).exists(): Path(output_path).unlink(missing_ok=True)
            error_detail = str(e).split('\n')[0][:200]
            return {"audio_path": None, "text": cleaned_text, "language": lang, "error": f"Synthesis failed ({error_detail})"}

    def _clean_text(self, text: str) -> str:
        # (Keep _clean_text method exactly as in Response #9)
        if not isinstance(text, str):
             logger.warning(f"Received non-string input for text cleaning: {type(text)}. Converting.")
             text = str(text)
        cleaned = re.sub(r'\s+', ' ', text).strip()
        # Keep basic punctuation addition if needed
        # if cleaned and not re.search(r'[.!?]$', cleaned):
        #     cleaned += '.'
        logger.debug(f"Cleaned text: '{text}' -> '{cleaned}'")
        return cleaned