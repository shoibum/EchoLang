import torch
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import os

from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode

class M4TASRModel:
    """
    Wrapper for Seamless M4T ASR model.
    """
    
    MODEL_URLS = {
        "seamless_m4t": "https://dl.fbaipublicfiles.com/seamless/models/seamless_m4t_medium.pt",
    }
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.device = self.model_manager.device
        self.model = None
        self.processor = None
        
    def load_model(self):
        """Load the Seamless M4T model."""
        try:
            # Ensure model file is downloaded
            model_path = self.model_manager.ensure_model_downloaded(
                "seamless_m4t_medium.pt", 
                self.MODEL_URLS["seamless_m4t"]
            )
            
            # Import here to avoid loading unnecessary dependencies if model not used
            from seamless_communication.models.inference import Translator
            
            # Load the M4T model
            self.model = Translator(
                model_name_or_path=str(model_path),
                vocoder_name_or_path=None,  # Will use default
                device=self.device,
                dtype=torch.float16 if self.device.type in ["cuda", "mps"] else torch.float32
            )
            
            print(f"Loaded Seamless M4T model on {self.device}")
            
        except Exception as e:
            print(f"Error loading Seamless M4T model: {e}")
            raise
    
    def transcribe(self, audio_path: Union[str, Path], 
                 src_lang: str = LanguageCode.ENGLISH,
                 return_timestamps: bool = False) -> Dict:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            src_lang: Source language code
            return_timestamps: Whether to return word-level timestamps
            
        Returns:
            Dictionary with transcription results
        """
        if self.model is None:
            self.load_model()
        
        # Convert language code to M4T format
        m4t_lang_code = LanguageCode.code_to_m4t_code(src_lang)
        
        # Process audio file using M4T
        text_output = self.model.predict(
            input=str(audio_path),
            task_str="s2tt",  # speech to transcription
            tgt_lang=m4t_lang_code,
            src_lang=m4t_lang_code,
        )
        
        result = {
            "text": text_output.translation,
            "language": src_lang,
        }
        
        if return_timestamps:
            # Note: M4T doesn't provide word-level timestamps directly
            # This is a placeholder for compatibility
            result["timestamps"] = []
            
        return result
