# src/tts/xtts.py (Allowlisting XttsArgs as well)
import torch
import os
import tempfile
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import re
import logging
import time
import torch.serialization # Import the serialization module

# Use relative imports
from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode
try:
    from .. import config as main_config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config as main_config

logger = logging.getLogger(__name__)

# --- Import ALL required classes for allowlisting ---
try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import XttsAudioConfig
    from TTS.config.shared_configs import BaseDatasetConfig
    # Import the FOURTH class mentioned in the error
    from TTS.tts.models.xtts import XttsArgs
    logger.debug("Successfully imported TTS components (TTS, XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs).")
except ImportError as e:
    logger.error("Failed to import TTS components. Is TTS installed correctly? (pip install TTS)", exc_info=True)
    # Set flags or raise error if imports fail
    XttsConfig = None
    XttsAudioConfig = None
    BaseDatasetConfig = None
    XttsArgs = None # Make sure this exists even if import fails
    raise ImportError("Could not import TTS components required for loading.") from e
# --- END Imports ---

class XTTSModel:
    """
    Wrapper for XTTS-v2 model using the high-level TTS.api interface.
    Handles model loading via API and synthesis using tts_to_file.
    FORCES CPU execution. Uses safe_globals context for loading.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        # (Keep __init__ method the same)
        self.model_manager = model_manager or ModelManager()
        self.device = "cpu"; self.tts_api: Optional[TTS] = None
        self.model_key = main_config.DEFAULT_XTTS_MODEL_KEY
        self.xtts_config = main_config.XTTS_V2_CONFIG.get(self.model_key)
        if not self.xtts_config: raise ValueError(f"Config not found for XTTS key '{self.model_key}'.")
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info(f"Initializing XTTSModel wrapper for API model: '{self.model_name}' (FORCING CPU execution)")

    def load_model(self):
        """Loads the XTTS model using TTS.api within safe_globals context."""
        if self.tts_api is not None: logger.debug("TTS API instance already loaded."); return
        # Check if classes needed for allowlisting were imported successfully
        if XttsConfig is None or XttsAudioConfig is None or BaseDatasetConfig is None or XttsArgs is None:
             raise RuntimeError("Cannot load XTTS model: Required TTS classes failed to import.")

        logger.info(f"Loading TTS API model: {self.model_name}"); logger.info(f"Forcing device: {self.device}")
        try:
            start_load_time = time.time()
            # --- MODIFIED: Add XttsArgs to the allowlist ---
            with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
                logger.debug("Attempting to load model inside safe_globals context for XttsConfig, XttsAudioConfig, BaseDatasetConfig, AND XttsArgs...")
                self.tts_api = TTS(model_name=self.model_name, progress_bar=True, gpu=False)
            # --- END MODIFICATION ---

            self.tts_api.to(self.device)
            end_load_time = time.time()
            logger.info(f"Successfully loaded TTS API model '{self.model_name}' onto CPU in {end_load_time - start_load_time:.2f}s.")
        except Exception as e:
            logger.error(f"Error loading TTS API model '{self.model_name}': {e}", exc_info=True)
            self.tts_api = None
            raise RuntimeError(f"Failed to load TTS API model '{self.model_name}': {e}") from e

    # --- Keep _get_reference_audio_path ---
    def _get_reference_audio_path(self, lang_code: str) -> Optional[Path]:
        local_dir_name = self.xtts_config.get("local_dir", "xtts_v2")
        speaker_dir = self.model_manager.get_model_path(local_dir_name) / "speakers"
        return speaker_dir / f"{lang_code}_reference.wav"

    # --- Keep _prepare_reference_audio ---
    def _prepare_reference_audio(self, lang_code: str) -> Optional[str]:
        ref_audio_path = self._get_reference_audio_path(lang_code)
        logger.debug(f"Looking for default speaker reference audio at: {ref_audio_path}")
        if ref_audio_path.exists(): logger.info(f"Using default reference audio: {ref_audio_path}"); return str(ref_audio_path)
        else: logger.error(f"Default reference speaker audio not found for '{lang_code}'. Expected: {ref_audio_path}."); return None

    # --- Keep synthesize ---
    def synthesize(self,
                   text: str,
                   lang: str = LanguageCode.ENGLISH,
                   speaker_audio: Optional[str] = None) -> Dict:
        # (Keep method as is - from Response #63)
        if self.tts_api is None:
            logger.info("TTS API instance not loaded. Calling load_model()...")
            try: self.load_model()
            except Exception as load_err: logger.error(f"Failed to load TTS API model: {load_err}", exc_info=True); return {"audio_path": None, "text": text, "language": lang, "error": f"Model load failed: {type(load_err).__name__}"}
            if self.tts_api is None: error_msg = "TTS API model could not be loaded."; logger.error(error_msg); return {"audio_path": None, "text": text, "language": lang, "error": error_msg}
        cleaned_text = self._clean_text(text)
        if not cleaned_text.strip(): logger.warning("Input text empty after cleaning."); return {"audio_path": None, "text": cleaned_text, "language": lang, "error": "Input text is empty"}
        speaker_wav_path = speaker_audio
        if speaker_wav_path:
             logger.info(f"Using provided speaker reference: {speaker_wav_path}")
             if not Path(speaker_wav_path).exists(): logger.error(f"Provided speaker audio not found: {speaker_wav_path}"); return {"audio_path": None, "text": cleaned_text, "language": lang, "error": "Provided speaker audio not found"}
        else:
             logger.info(f"No speaker reference provided. Attempting default for language: {lang}")
             speaker_wav_path = self._prepare_reference_audio(lang)
             if speaker_wav_path is None:
                  supported_langs_no_default = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh-cn', 'hu', 'ko', 'ja', 'hi']
                  if lang in supported_langs_no_default: logger.error(f"Cannot synth: No speaker ref and default for '{lang}' missing."); return {"audio_path": None, "text": cleaned_text, "language": lang, "error": f"Speaker reference required for '{lang}' (default missing)"}
        xtts_lang = lang.lower(); logger.debug(f"Using language code for TTS API: '{xtts_lang}'")
        output_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file: output_path = tmp_file.name
            logger.debug(f"Created temporary output file: {output_path}")
            start_time = time.time()
            logger.info(f"Starting XTTS synthesis (CPU): '{cleaned_text[:50]}...', Lang: {xtts_lang}")
            self.tts_api.tts_to_file( text=cleaned_text, speaker_wav=speaker_wav_path, language=xtts_lang, file_path=output_path, split_sentences=True )
            logger.info(f"XTTS synthesis successful in {time.time() - start_time:.2f}s. Output: {output_path}")
            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                 logger.error("TTS output file missing or empty.")
                 if Path(output_path).exists():
                     try: os.unlink(output_path)
                     except OSError as unlink_err: logger.warning(f"Could not delete potentially empty temp file {output_path}: {unlink_err}")
                 return {"audio_path": None, "text": cleaned_text, "language": lang, "error": "Synthesis failed (empty output file)"}
            return {"audio_path": output_path, "text": cleaned_text, "language": lang, "error": None }
        except Exception as e:
            if f"Language {lang} is not supported" in str(e): logger.error(f"XTTS API reported lang '{lang}' not supported: {e}"); error_detail = f"XTTS does not support language '{lang}'"
            elif "Unsupported global" in str(e) or "WeightsUnpickler" in str(e): logger.error(f"Unpickler error during synthesis call: {e}", exc_info=True); error_detail=f"Model loading/unpickling error ({type(e).__name__})"
            else: logger.error(f"Error during TTS API synthesis: {e}", exc_info=True); error_detail = f"Synthesis failed ({str(e).splitlines()[0][:100]})"
            if output_path and Path(output_path).exists():
                 try: os.unlink(output_path); logger.info(f"Cleaned up temp file on error: {output_path}")
                 except OSError as unlink_err: logger.warning(f"Could not delete temp file {output_path} on error: {unlink_err}")
            return {"audio_path": None, "text": cleaned_text, "language": lang, "error": error_detail}

    # --- Keep _clean_text ---
    def _clean_text(self, text: str) -> str:
        # (Keep method as is)
        if not isinstance(text, str): logger.warning(f"Non-string input: {type(text)}. Converting."); text = str(text)
        cleaned = re.sub(r'\s+', ' ', text).strip()
        logger.debug(f"Cleaned text: '{text}' -> '{cleaned}'")
        return cleaned