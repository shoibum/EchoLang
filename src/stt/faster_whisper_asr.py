# src/stt/faster_whisper_asr.py
import logging
from typing import Optional, Dict, Union
from pathlib import Path
import time
import itertools

# Use relative imports for utils and config
from ..utils.model_utils import ModelManager # May not be needed if config handles all
from ..utils.language import LanguageCode
try:
    from .. import config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    # from faster_whisper.utils import format_timestamp # If timestamp formatting needed
    logger.debug("Successfully imported faster_whisper.")
except ImportError as e:
    logger.error("Failed to import faster-whisper. Is it installed? (pip install faster-whisper)", exc_info=True)
    # Set WhisperModel to None or raise specific error if critical
    WhisperModel = None
    # raise ImportError("Could not import faster_whisper.") from e

class FasterWhisperASRModel:
    """
    Wrapper for FasterWhisper model using CTranslate2 backend.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        if WhisperModel is None:
             raise RuntimeError("faster-whisper library is not installed or failed to import.")

        # Get config details from central config.py
        self.model_key = config.DEFAULT_FASTER_WHISPER_MODEL_KEY
        model_config = config.FASTER_WHISPER_CONFIG.get(self.model_key)
        if not model_config:
            raise ValueError(f"Config not found for faster-whisper model key '{self.model_key}' in config.FASTER_WHISPER_CONFIG.")

        self.model_size = model_config.get("model_size_or_path")
        self.device = model_config.get("device")
        self.compute_type = model_config.get("compute_type")
        self.model: Optional[WhisperModel] = None # Lazy load model

        if not self.model_size:
            raise ValueError(f"Missing 'model_size_or_path' in FASTER_WHISPER_CONFIG for key '{self.model_key}'.")

        logger.info(f"Initializing FasterWhisperASRModel: size='{self.model_size}', target_device='{self.device}', compute_type='{self.compute_type}'")

    def load_model(self):
        """Loads the FasterWhisper model. Conversion happens automatically on first load."""
        if self.model is not None:
            logger.debug("FasterWhisper model already loaded.")
            return

        logger.info(f"Loading faster-whisper model: '{self.model_size}'...")
        logger.info(f"Will use device='{self.device}' with compute_type='{self.compute_type}'.")
        # Warn if using CPU fallback from MPS/CUDA intention
        if config.APP_DEVICE != self.device and self.device == "cpu":
             logger.warning(f"Configured faster-whisper device is '{self.device}', but intended app device was '{config.APP_DEVICE}'. Using CPU.")
        elif config.APP_DEVICE == "mps" and self.device == "mps":
             logger.info("Attempting to use MPS device for faster-whisper.")

        try:
            # Download and conversion to CTranslate2 format happens automatically
            # by faster-whisper if the model size isn't found in cache.
            # Cache location usually ~/.cache/huggingface/hub/models--guillaumekln--faster-whisper-<model_size>
            start_load_time = time.time()
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
                # cpu_threads=4, # Optional: Set number of threads for CPU
                # num_workers=1, # Optional: For multi-GPU dataloading
            )
            end_load_time = time.time()
            logger.info(f"Faster-whisper model '{self.model_size}' loaded successfully in {end_load_time - start_load_time:.2f}s.")

        except Exception as e:
            logger.error(f"Error loading faster-whisper model '{self.model_size}': {e}", exc_info=True)
            self.model = None
            # Provide specific hints if possible
            if "mps" in str(e).lower():
                 logger.error("Loading failed. This might indicate an MPS incompatibility with CTranslate2/FasterWhisper.")
                 logger.error("Consider changing 'device' to 'cpu' in FASTER_WHISPER_CONFIG in config.py and restarting.")
            elif "module 'ctranslate2' has no attribute 'Device'" in str(e) or "CTranslate2" in str(e):
                 logger.error("CTranslate2 library might be missing or corrupted. Try reinstalling: pip install -U ctranslate2 faster-whisper")

            # Re-raise or handle as appropriate for the application lifecycle
            raise RuntimeError(f"Failed to load faster-whisper model: {e}") from e

    def transcribe(self, audio_path: Union[str, Path],
                   src_lang: Optional[str] = None) -> Dict:
        """
        Transcribe audio file using FasterWhisper.

        Args:
            audio_path: Path to audio file.
            src_lang: Source language code (e.g., 'en', 'hi', 'kn').
                      Passed as language hint. None for auto-detect.

        Returns:
            Dictionary with transcription results {'text': str, 'language': str}
        """
        if self.model is None:
            logger.info("Faster-whisper model not loaded. Calling load_model()...")
            try:
                self.load_model()
            except Exception as load_err:
                 logger.error(f"Failed to load faster-whisper model during transcribe call: {load_err}", exc_info=True)
                 return {"text": f"ERROR: Model load failed - {type(load_err).__name__}", "language": src_lang or "unknown"}
            if self.model is None: # Should not happen if load_model raises error, but check anyway
                 error_msg = "Faster-whisper model could not be loaded. Cannot transcribe."
                 logger.error(error_msg)
                 return {"text": "ERROR: Model not loaded", "language": src_lang or "unknown"}

        audio_path_str = str(audio_path)
        # Faster-whisper uses 2-letter codes, or None for auto-detect. Our internal codes match.
        language_hint = src_lang

        logger.info(f"Starting faster-whisper transcription for: {audio_path_str}, Language hint: {language_hint}")

        if not Path(audio_path_str).is_file():
             error_msg = f"Audio file not found: {audio_path_str}"
             logger.error(error_msg)
             return {"text": f"ERROR: Audio file not found", "language": src_lang or "unknown"}

        try:
            start_time = time.time()
            # See faster-whisper docs for more options: beam_size, vad_filter, temperature, etc.
            # Using VAD filter can improve accuracy on long files with silence.
            segments, info = self.model.transcribe(
                audio_path_str,
                language=language_hint,
                beam_size=5,
                vad_filter=True, # Enable VAD filter
                vad_parameters=dict(min_silence_duration_ms=500), # Default VAD params
                # task="transcribe", # Default is transcribe
                # temperature=0.0, # Default temperature
            )

            # Concatenate text from all segments
            # Need to handle the generator properly
            # Use itertools.chain to handle potential StopIteration if segments is empty
            full_text = "".join(s.text for s in segments).strip()

            end_time = time.time()

            detected_language = info.language
            lang_prob = info.language_probability
            duration = info.duration

            logger.info(f"Faster-whisper transcription successful in {end_time - start_time:.2f}s (Audio duration: {duration:.2f}s).")
            logger.info(f"Detected language: {detected_language} (Probability: {lang_prob:.2f})")
            logger.info(f"Result: '{full_text[:150]}...'")

            # Return format consistent with previous STT implementation
            # Use detected language if available, otherwise fall back to hint or unknown
            final_lang = detected_language if detected_language else (src_lang or "unknown")

            # Transliteration logic is removed here. Re-add if needed.

            return {
                "text": full_text,
                "language": final_lang
            }

        except Exception as e:
            logger.error(f"Error during faster-whisper transcription process for {audio_path_str}: {e}", exc_info=True)
            # Check for specific CTranslate2/MPS errors if possible
            error_detail = str(e).split('\n')[0][:200]
            return {"text": f"ERROR: Transcription failed ({error_detail})", "language": src_lang or "unknown"}