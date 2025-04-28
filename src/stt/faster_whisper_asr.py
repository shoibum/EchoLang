# src/stt/faster_whisper_asr.py
import logging
from typing import Optional, Dict, Union
from pathlib import Path
import time
import itertools

# Removed relative config import - config passed directly
# Use relative imports for utils
from ..utils.model_utils import ModelManager # May not be needed if config handles all
from ..utils.language import LanguageCode
# Import main config to check APP_DEVICE for warning
try:
    from .. import config as main_config
except ImportError:
    # Handle running file directly if necessary, though unlikely in final app
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config as main_config


logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    logger.debug("Successfully imported faster_whisper.")
except ImportError as e:
    logger.error("Failed to import faster-whisper. Is it installed? (pip install faster-whisper)", exc_info=True)
    WhisperModel = None

class FasterWhisperASRModel:
    """
    Wrapper for a specific FasterWhisper model using CTranslate2 backend.
    Loads model based on provided configuration.
    """

    # MODIFIED: __init__ now takes a specific model_config dictionary
    def __init__(self, model_config: Dict, model_key: str):
        if WhisperModel is None:
             raise RuntimeError("faster-whisper library is not installed or failed to import.")

        self.model_key = model_key # Store the key for logging
        self.model_config = model_config

        # Get details from the provided config dictionary
        self.model_path = self.model_config.get("model_path") # Expecting path now
        self.device = self.model_config.get("device")
        self.compute_type = self.model_config.get("compute_type")
        self.model: Optional[WhisperModel] = None # Lazy load model

        if not self.model_path:
            raise ValueError(f"Missing 'model_path' in FASTER_WHISPER_CONFIG for key '{self.model_key}'.")

        logger.info(f"Initializing FasterWhisperASRModel '{self.model_key}': path='{self.model_path}', target_device='{self.device}', compute_type='{self.compute_type}'")

    def load_model(self):
        """Loads the FasterWhisper model from the specified path."""
        if self.model is not None:
            logger.debug(f"FasterWhisper model '{self.model_key}' already loaded.")
            return

        logger.info(f"Loading faster-whisper model '{self.model_key}' from path: '{self.model_path}'...")
        logger.info(f"Will use device='{self.device}' with compute_type='{self.compute_type}'.")

        # Check if the model path exists
        if not Path(self.model_path).exists():
             error_msg = f"Model directory not found for key '{self.model_key}': {self.model_path}. Did the conversion complete successfully?"
             logger.error(error_msg)
             raise FileNotFoundError(error_msg)
        if not (Path(self.model_path) / "model.bin").exists():
            error_msg = f"'model.bin' not found in directory for key '{self.model_key}': {self.model_path}. Is this a correctly converted CTranslate2 model directory?"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Warn if using CPU fallback from MPS/CUDA intention (using main config APP_DEVICE)
        # Note: This check might be less relevant now if all models run on CPU anyway
        if main_config.APP_DEVICE != self.device and self.device == "cpu":
             logger.warning(f"Configured faster-whisper device for '{self.model_key}' is '{self.device}', but intended app device was '{main_config.APP_DEVICE}'. Using CPU.")
        elif main_config.APP_DEVICE == "mps" and self.device == "mps":
             logger.info(f"Attempting to use MPS device for faster-whisper model '{self.model_key}'.")

        try:
            # Load the CTranslate2 model directly from the path
            start_load_time = time.time()
            self.model = WhisperModel(
                self.model_path, # Load from directory path
                device=self.device,
                compute_type=self.compute_type
            )
            end_load_time = time.time()
            logger.info(f"Faster-whisper model '{self.model_key}' loaded successfully from {self.model_path} in {end_load_time - start_load_time:.2f}s.")

        except Exception as e:
            logger.error(f"Error loading faster-whisper model '{self.model_key}' from {self.model_path}: {e}", exc_info=True)
            self.model = None
            if "mps" in str(e).lower():
                 logger.error("Loading failed. This might indicate an MPS incompatibility with CTranslate2/FasterWhisper.")
                 logger.error("Consider changing 'device' to 'cpu' in FASTER_WHISPER_CONFIG in config.py and restarting.")
            elif "CTranslate2" in str(e):
                 logger.error("CTranslate2 library might be missing or corrupted. Try reinstalling: pip install -U ctranslate2 faster-whisper")

            raise RuntimeError(f"Failed to load faster-whisper model '{self.model_key}': {e}") from e

    def transcribe(self, audio_path: Union[str, Path],
                   src_lang: Optional[str] = None) -> Dict:
        """
        Transcribe audio file using the loaded FasterWhisper model instance.

        Args:
            audio_path: Path to audio file.
            src_lang: Source language code (e.g., 'en', 'hi', 'kn').
                      Passed as language hint. None for auto-detect.

        Returns:
            Dictionary with transcription results {'text': str, 'language': str}
        """
        if self.model is None:
            logger.info(f"Faster-whisper model '{self.model_key}' not loaded. Calling load_model()...")
            try:
                self.load_model()
            except Exception as load_err:
                 logger.error(f"Failed to load faster-whisper model '{self.model_key}' during transcribe call: {load_err}", exc_info=True)
                 return {"text": f"ERROR: Model '{self.model_key}' load failed - {type(load_err).__name__}", "language": src_lang or "unknown"}
            if self.model is None: # Should not happen if load_model raises error, but check anyway
                 error_msg = f"Faster-whisper model '{self.model_key}' could not be loaded. Cannot transcribe."
                 logger.error(error_msg)
                 return {"text": f"ERROR: Model '{self.model_key}' not loaded", "language": src_lang or "unknown"}

        audio_path_str = str(audio_path)
        language_hint = src_lang

        logger.info(f"Starting transcription with model '{self.model_key}' for: {audio_path_str}, Language hint: {language_hint}")

        if not Path(audio_path_str).is_file():
             error_msg = f"Audio file not found: {audio_path_str}"
             logger.error(error_msg)
             return {"text": f"ERROR: Audio file not found", "language": src_lang or "unknown"}

        try:
            start_time = time.time()
            segments, info = self.model.transcribe(
                audio_path_str,
                language=language_hint,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            # Concatenate text from all segments
            full_text = "".join(s.text for s in segments).strip()
            end_time = time.time()

            detected_language = info.language
            lang_prob = info.language_probability
            duration = info.duration

            logger.info(f"Model '{self.model_key}' transcription successful in {end_time - start_time:.2f}s (Audio duration: {duration:.2f}s).")
            logger.info(f"Detected language: {detected_language} (Probability: {lang_prob:.2f})")
            # Limit log length
            log_text = full_text[:150].replace('\n', ' ') + ('...' if len(full_text) > 150 else '')
            logger.info(f"Result: '{log_text}'")


            final_lang = detected_language if detected_language else (src_lang or "unknown")

            return {
                "text": full_text,
                "language": final_lang
            }

        except Exception as e:
            logger.error(f"Error during transcription with model '{self.model_key}' for {audio_path_str}: {e}", exc_info=True)
            error_detail = str(e).split('\n')[0][:200]
            return {"text": f"ERROR: Transcription failed ({error_detail})", "language": src_lang or "unknown"}