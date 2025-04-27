# src/tts/xtts.py
import torch
import os
import tempfile
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import re
import logging
import time
# soundfile is no longer needed here as tts_to_file handles saving
# import soundfile as sf
# import numpy as np

# Use relative imports for utils and config
from ..utils.model_utils import ModelManager # Still needed for speaker path
from ..utils.language import LanguageCode
# from ..utils.audio import AudioProcessor # Likely not needed with API wrapper
try:
    from .. import config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

# --- CHANGE: Import the high-level API wrapper ---
try:
    from TTS.api import TTS
    logger.debug("Successfully imported TTS.api.TTS.")
except ImportError as e:
    logger.error("Failed to import TTS.api. Is TTS installed correctly? (pip install TTS)", exc_info=True)
    raise ImportError("Could not import TTS.api.") from e
# --- No longer need direct XttsConfig or Xtts model import ---
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import Xtts
# from TTS.tts.utils.speakers import SpeakerManager

class XTTSModel:
    """
    Wrapper for XTTS-v2 model using the high-level TTS.api interface.
    Handles model loading via API and synthesis using tts_to_file.
    FORCES CPU execution due to observed MPS incompatibilities.
    """
    # OUTPUT_SAMPLE_RATE = 24000 # API handles sample rate

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        # --- CHANGE: Force CPU for XTTS ---
        self.device = "cpu" # Override config.APP_DEVICE for XTTS
        self.tts_api: Optional[TTS] = None # API instance loaded on demand

        self.model_key = config.DEFAULT_XTTS_MODEL_KEY # Used to get model name
        self.xtts_config = config.XTTS_V2_CONFIG.get(self.model_key) # Still useful for speaker path
        if not self.xtts_config:
            raise ValueError(f"Configuration for XTTS model key '{self.model_key}' not found in config.py.")

        # --- CHANGE: Use the standard model name for the API ---
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        # --- Updated log message ---
        logger.info(f"Initializing XTTSModel wrapper for API model: '{self.model_name}' (FORCING CPU execution)")

    def load_model(self):
        """Load the XTTS-v2 model using the TTS.api interface onto the CPU."""
        if self.tts_api is not None:
            logger.debug("TTS API instance already loaded.")
            return

        logger.info(f"Loading TTS API model: {self.model_name}")
        # --- Updated log message ---
        logger.info(f"Forcing device: {self.device} (Ignoring config.APP_DEVICE for XTTS)")

        try:
            start_load_time = time.time()
            # --- CHANGE: Initialize using TTS(model_name) ---
            # The API handles finding/downloading the model specified by name
            # Force gpu=False to ensure CPU loading even if CUDA is available elsewhere
            self.tts_api = TTS(model_name=self.model_name, progress_bar=True, gpu=False)
            # --- CHANGE: Explicitly move to CPU (should be default if gpu=False, but enforce) ---
            self.tts_api.to(self.device)
            end_load_time = time.time()
            logger.info(f"Successfully loaded TTS API model '{self.model_name}' onto CPU in {end_load_time - start_load_time:.2f}s.")

        except Exception as e:
            logger.error(f"Error loading TTS API model '{self.model_name}': {e}", exc_info=True)
            self.tts_api = None
            raise RuntimeError(f"Failed to load TTS API model '{self.model_name}': {e}") from e

    def _get_reference_audio_path(self, lang_code: str) -> Optional[Path]:
        """Construct the expected path for the default reference audio."""
        # Use local_dir from config associated with the model key
        local_dir_name = self.xtts_config.get("local_dir", "xtts_v2")
        speaker_dir = self.model_manager.get_model_path(local_dir_name) / "speakers"
        ref_audio_path = speaker_dir / f"{lang_code}_reference.wav"
        return ref_audio_path

    def _prepare_reference_audio(self, lang_code: str) -> Optional[str]:
        """
        Get the path to the reference audio for the given language.
        Logs an error if the default file is not found. Returns None if missing.
        """
        ref_audio_path = self._get_reference_audio_path(lang_code)
        logger.debug(f"Looking for default speaker reference audio at: {ref_audio_path}")

        if ref_audio_path.exists():
            logger.info(f"Using default reference audio: {ref_audio_path}")
            return str(ref_audio_path)
        else:
            logger.error(f"Default reference speaker audio not found for language '{lang_code}'. Expected at: {ref_audio_path}. Cannot use default speaker.")
            return None # Let synthesize handle missing speaker

    def synthesize(self,
                   text: str,
                   lang: str = LanguageCode.ENGLISH,
                   speaker_audio: Optional[str] = None,
                   speed: float = 1.0) -> Dict:
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
                  logger.error(f"Cannot synthesize: No speaker reference provided and default speaker for '{lang}' is missing.")
                  return {"audio_path": None, "text": cleaned_text, "language": lang, "error": f"Default speaker for '{lang}' not found"}

        # Language code for XTTS API should be the simple 2-letter code
        xtts_lang = lang.lower()
        logger.debug(f"Using language code for TTS API: '{xtts_lang}'")

        output_path = None
        try:
            # Create a temporary file path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                 output_path = tmp_file.name
            logger.debug(f"Created temporary output file: {output_path}")

            start_time = time.time()
            logger.info(f"Starting XTTS synthesis via API (CPU) for text: '{cleaned_text[:50]}...', Lang: {xtts_lang}, Speed: {speed}")

            # Use tts_api.tts_to_file
            self.tts_api.tts_to_file(
                text=cleaned_text,
                speaker_wav=speaker_wav_path,
                language=xtts_lang,
                file_path=output_path,
                speed=speed, # Pass speed parameter
                split_sentences=True # Good default for longer text
            )

            end_time = time.time()
            logger.info(f"XTTS synthesis via API (CPU) successful in {end_time - start_time:.2f}s. Output: {output_path}")

            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                 logger.error("TTS synthesis seemed to succeed but output file is missing or empty.")
                 if Path(output_path).exists():
                     try: Path(output_path).unlink()
                     except OSError: pass
                 return {"audio_path": None, "text": cleaned_text, "language": lang, "error": "Synthesis failed (empty output file)"}

            return {
                "audio_path": output_path,
                "text": cleaned_text, # Return the input text for consistency
                "language": lang,
                "error": None
            }

        except FileNotFoundError as e:
             logger.error(f"File not found during synthesis: {e}", exc_info=True)
             if output_path and Path(output_path).exists(): Path(output_path).unlink(missing_ok=True)
             return {"audio_path": None, "text": cleaned_text, "language": lang, "error": f"File not found: {e.filename}"}
        except Exception as e:
            logger.error(f"Error during TTS API synthesis process: {e}", exc_info=True)
            if output_path and Path(output_path).exists(): Path(output_path).unlink(missing_ok=True)
            error_detail = str(e).split('\n')[0][:200]
            return {"audio_path": None, "text": cleaned_text, "language": lang, "error": f"Synthesis failed ({error_detail})"}

    def _clean_text(self, text: str) -> str:
        """Clean text for XTTS synthesis."""
        if not isinstance(text, str):
             logger.warning(f"Received non-string input for text cleaning: {type(text)}. Converting.")
             text = str(text)

        cleaned = re.sub(r'\s+', ' ', text).strip()
        if cleaned and not re.search(r'[.!?]$', cleaned):
            cleaned += '.'
        logger.debug(f"Cleaned text: '{text}' -> '{cleaned}'")
        return cleaned