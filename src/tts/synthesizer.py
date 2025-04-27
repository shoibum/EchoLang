# src/tts/synthesizer.py
from typing import Optional, Dict, Union
from pathlib import Path

from .xtts import XTTSModel
from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode

class TextToSpeech:
    """
    Text-to-speech interface for EchoLang.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.tts_model = XTTSModel(self.model_manager)

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
            Dictionary with synthesis results {'audio_path': str|None, 'text': str, 'language': str, 'error': str|None}
        """
        return self.tts_model.synthesize(
            text=text,
            lang=lang,
            speaker_audio=speaker_audio,
            speed=speed
        )