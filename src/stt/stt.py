# src/stt/stt.py
from typing import Optional, Dict, Union
from pathlib import Path

from .m4t_asr import M4TASRModel
from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode

class SpeechToText:
    """
    Speech-to-text interface for EchoLang.
    """
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.asr_model = M4TASRModel(self.model_manager)
        
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
        return self.asr_model.transcribe(
            audio_path=audio_path,
            src_lang=src_lang,
            return_timestamps=return_timestamps
        )