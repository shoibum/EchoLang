# src/tts/mms_tts.py
import torch
import scipy.io.wavfile
import tempfile
from pathlib import Path
import logging
from typing import Optional, Dict

# Use relative imports for config
try:
    from .. import config
except ImportError:
    import sys
    # Allow running directly for testing? Unlikely needed if main.py is entry point
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

try:
    from transformers import VitsModel, AutoTokenizer
    logger.debug("Successfully imported transformers components for MMS TTS.")
except ImportError as e:
    logger.error("Failed to import transformers. Is it installed correctly?", exc_info=True)
    # Set models to None or raise specific error if critical
    VitsModel = None
    AutoTokenizer = None


class MMS_TTSModel:
    """
    Wrapper for Facebook's MMS-TTS model (specifically for Kannada).
    Uses Hugging Face transformers library. FORCES CPU EXECUTION.
    """
    MODEL_ID = "facebook/mms-tts-kan" # Kannada model

    def __init__(self):
        if VitsModel is None or AutoTokenizer is None:
            raise RuntimeError("transformers library is not installed or failed to import.")

        # --- FORCE CPU FOR THIS MODEL ---
        self.device = "cpu"
        # --- END CHANGE ---
        # Dtype will be float32 for CPU
        self.torch_dtype = torch.float32

        self.model: Optional[VitsModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_loaded = False

        logger.info(f"Initializing MMS_TTSModel wrapper for: {self.MODEL_ID}")
        # Log the forced device
        logger.info(f"FORCING CPU execution for this model (device={self.device}, dtype={self.torch_dtype})")


    def load_model(self):
        """Loads the MMS-TTS model and tokenizer onto the CPU."""
        if self.model_loaded:
            logger.debug(f"MMS-TTS model {self.MODEL_ID} already loaded.")
            return

        logger.info(f"Loading MMS-TTS model: {self.MODEL_ID} onto CPU...")
        load_dtype = self.torch_dtype # Should be float32
        # self.device is already set to "cpu" in __init__

        try:
            # Load model and tokenizer, ensuring model is on CPU
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
            # Explicitly load model to CPU with correct dtype
            self.model = VitsModel.from_pretrained(self.MODEL_ID).to(dtype=load_dtype, device=self.device)
            self.model_loaded = True
            logger.info(f"Successfully loaded MMS-TTS model {self.MODEL_ID} onto {self.device}.")

        except Exception as e:
            logger.error(f"Error loading MMS-TTS model {self.MODEL_ID}: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            self.model_loaded = False
            raise RuntimeError(f"Failed to load MMS-TTS model {self.MODEL_ID}") from e

    def synthesize(self, text: str) -> Dict:
        """
        Synthesize speech from Kannada text using MMS-TTS on CPU.

        Args:
            text: Kannada text to synthesize.

        Returns:
            Dictionary with synthesis results {'audio_path': str|None, 'text': str, 'language': str, 'error': str|None}
        """
        if not self.model_loaded:
            logger.info("MMS-TTS model not loaded. Calling load_model()...")
            try:
                 self.load_model()
            except Exception as load_err:
                 logger.error(f"Failed to load MMS-TTS model during synthesize call: {load_err}", exc_info=True)
                 return {"audio_path": None, "text": text, "language": "kn", "error": f"Model load failed: {type(load_err).__name__}"}
            if not self.model_loaded:
                 error_msg = "MMS-TTS model could not be loaded. Cannot synthesize."
                 logger.error(error_msg)
                 return {"audio_path": None, "text": text, "language": "kn", "error": error_msg}

        # Ensure tokenizer is loaded
        if self.tokenizer is None or self.model is None:
             error_msg = "MMS-TTS model or tokenizer is None after load attempt. Cannot synthesize."
             logger.error(error_msg)
             return {"audio_path": None, "text": text, "language": "kn", "error": error_msg}


        # MMS specific synthesis steps
        output_path = None
        try:
            # Log using self.device which is now 'cpu'
            logger.info(f"Starting MMS-TTS synthesis ({self.device}) for text: '{text[:50]}...'")
            # Tokenizer runs on CPU, inputs stay on CPU
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device) # Should be CPU

            with torch.no_grad():
                output_waveform = self.model(**inputs).waveform

            if output_waveform is None or output_waveform.numel() == 0:
                 logger.error("MMS-TTS synthesis resulted in empty waveform.")
                 return {"audio_path": None, "text": text, "language": "kn", "error": "Synthesis failed (empty waveform)"}

            # Create a temporary file path
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                 output_path = tmp_file.name
            logger.debug(f"Created temporary output file for MMS-TTS: {output_path}")

            # Save the waveform (already on CPU)
            waveform_np = output_waveform.squeeze().detach().numpy().astype('float32')
            sampling_rate = self.model.config.sampling_rate
            scipy.io.wavfile.write(output_path, rate=sampling_rate, data=waveform_np)

            logger.info(f"MMS-TTS synthesis successful. Output: {output_path}")

            if not Path(output_path).exists() or Path(output_path).stat().st_size == 0:
                 logger.error("MMS-TTS synthesis seemed to succeed but output file is missing or empty.")
                 if Path(output_path).exists(): Path(output_path).unlink(missing_ok=True)
                 return {"audio_path": None, "text": text, "language": "kn", "error": "Synthesis failed (empty output file)"}

            return {
                "audio_path": output_path,
                "text": text,
                "language": "kn", # Hardcoded as this model is for Kannada
                "error": None
            }

        except Exception as e:
            logger.error(f"Error during MMS-TTS synthesis process: {e}", exc_info=True)
            if output_path and Path(output_path).exists(): Path(output_path).unlink(missing_ok=True)
            error_detail = str(e).split('\n')[0][:200]
            return {"audio_path": None, "text": text, "language": "kn", "error": f"Synthesis failed ({error_detail})"}