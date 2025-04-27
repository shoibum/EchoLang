# src/tts/xtts.py
import torch
import os
import tempfile
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import re

from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode
from ..utils.audio import AudioProcessor

class XTTSModel:
    """
    Wrapper for XTTS-v2 model.
    """
    
    MODEL_URLS = {
        "xtts_v2": "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth",
        "xtts_v2_config": "https://huggingface.co/coqui/XTTS-v2/raw/main/config.json",
        "xtts_v2_vocab": "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json",
    }
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.device = self.model_manager.device
        self.model = None
        self.audio_processor = AudioProcessor()
        self.default_speakers = {
            LanguageCode.ENGLISH: "default_en",
            LanguageCode.HINDI: "default_hi",
            LanguageCode.KANNADA: "default_kn",
        }
        
    def load_model(self):
        """Load the XTTS-v2 model."""
        try:
            # Ensure model files are downloaded
            model_path = self.model_manager.ensure_model_downloaded(
                "xtts_v2/model.pth", 
                self.MODEL_URLS["xtts_v2"]
            )
            config_path = self.model_manager.ensure_model_downloaded(
                "xtts_v2/config.json", 
                self.MODEL_URLS["xtts_v2_config"]
            )
            vocab_path = self.model_manager.ensure_model_downloaded(
                "xtts_v2/vocab.json", 
                self.MODEL_URLS["xtts_v2_vocab"]
            )
            
            # Import here to avoid loading unnecessary dependencies if model not used
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
            
            # Load the XTTS model
            config = XttsConfig()
            config.load_json(str(config_path))
            
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(self.model_manager, str(model_path))
            
            if self.device.type == "mps":
                # For MPS, use mixed precision
                self.model.to(torch.device("mps"))
            else:
                self.model.to(self.device)
            
            print(f"Loaded XTTS-v2 model on {self.device}")
            
        except Exception as e:
            print(f"Error loading XTTS-v2 model: {e}")
            raise
    
    def _prepare_reference_audio(self, lang_code: str) -> str:
        """Get or create reference audio for the given language."""
        # In a real implementation, we would have reference files for each language
        # For simplicity in this example, we'll use the same speaker reference for all languages
        ref_audio_path = self.model_manager.get_model_path(f"xtts_v2/speakers/{lang_code}_reference.wav")
        
        if ref_audio_path.exists():
            return str(ref_audio_path)
        
        # In a real implementation, you would have pre-recorded reference audio files
        # This is just a placeholder - in real code, ensure reference audios are available
        ref_audio_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Create a dummy reference file for demo purposes
        import numpy as np
        dummy_audio = np.random.uniform(-0.1, 0.1, 16000*3)  # 3 seconds of random noise
        self.audio_processor.save_audio(
            torch.from_numpy(dummy_audio), 
            str(ref_audio_path), 
            sample_rate=16000
        )
        
        return str(ref_audio_path)
    
    def synthesize(self, 
                 text: str, 
                 lang: str = LanguageCode.ENGLISH,
                 speaker_audio: Optional[str] = None,
                 speed: float = 1.0) -> Dict:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            lang: Language code
            speaker_audio: Path to speaker reference audio (optional)
            speed: Speech speed factor
            
        Returns:
            Dictionary with synthesis results
        """
        if self.model is None:
            self.load_model()
            
        # Clean text for synthesis
        text = self._clean_text(text)
        if not text.strip():
            text = "Hello."  # Fallback if text is empty
        
        # Get reference audio
        speaker_wav = speaker_audio or self._prepare_reference_audio(lang)
        
        # Get language code in XTTS format
        xtts_lang = LanguageCode.code_to_xtts_locale(lang)
        
        # Use temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
            
        # Generate speech
        self.model.synthesize(
            text=text,
            language=xtts_lang,
            speaker_wav=speaker_wav,
            output_path=output_path,
            speed=speed
        )
        
        return {
            "audio_path": output_path,
            "text": text,
            "language": lang
        }
        
    def _clean_text(self, text: str) -> str:
        """Clean text for XTTS synthesis."""
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Ensure text ends with a sentence-ending punctuation
        if text and not re.search(r'[.!?]$', text):
            text += '.'
        return text