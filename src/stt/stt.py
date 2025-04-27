# src/stt/stt.py
from typing import Optional, Dict, Union
from pathlib import Path

# Updated import to use the new FasterWhisper wrapper
from .faster_whisper_asr import FasterWhisperASRModel
from ..utils.model_utils import ModelManager # Still useful potentially for other things
from ..utils.language import LanguageCode

class SpeechToText:
    """
    Speech-to-text interface for EchoLang (using FasterWhisper).
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        # Pass model_manager if FasterWhisperASRModel uses it, otherwise it can be None
        self.model_manager = model_manager # or ModelManager()
        # Instantiate the FasterWhisper model wrapper
        self.asr_model = FasterWhisperASRModel(self.model_manager)

    def transcribe(self, audio_path: Union[str, Path],
                   src_lang: Optional[str] = None) -> Dict:
        """
        Transcribe audio file to text using FasterWhisper.

        Args:
            audio_path: Path to audio file
            src_lang: Source language code hint (optional for FasterWhisper)

        Returns:
            Dictionary with transcription results {'text': str, 'language': str}
        """
        return self.asr_model.transcribe(
            audio_path=audio_path,
            src_lang=src_lang
        )