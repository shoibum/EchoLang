"""
Pipeline module for connecting STT, Translation, and TTS modules.
"""

from typing import Dict, Optional, Tuple, Union, BinaryIO
import os
import tempfile

from src.stt.stt import SpeechToText
from src.tts.tts import TextToSpeech
from src.translation.translator import Translator
from src.utils.audio import convert_audio_bytes_to_wav
from src.config import LANGUAGES

class Pipeline:
    """
    Pipeline class for connecting STT, Translation, and TTS modules.
    """
    
    def __init__(self):
        """Initialize the pipeline module."""
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.translator = Translator()
    
    def speech_to_text(
        self,
        audio: Union[str, bytes, BinaryIO],
        language: Optional[str] = None
    ) -> Dict:
        """
        Convert speech to text.
        Supports:
         - file path (str)
         - raw bytes (bytes)
         - file‐like (BinaryIO)
        """
        # 1) Raw bytes → use the bytes‐aware entrypoint
        if isinstance(audio, (bytes, bytearray)):
            return self.stt.transcribe_audio_bytes(audio, language)

        # 2) Otherwise delegate directly
        return self.stt.transcribe(audio, language)
    
    def text_to_speech(
        self, 
        text: str,
        language: str,
        output_path: Optional[str] = None
    ) -> Union[str, bytes]:
        """
        Convert text to speech.
        
        Args:
            text: Text to synthesize
            language: Target language code
            output_path: Path to save audio file (optional)
            
        Returns:
            Path to audio file or audio bytes
        """
        return self.tts.synthesize(text, language, output_path)
    
    def translate_text(
        self, 
        text: str,
        source_language: str,
        target_language: str
    ) -> str:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_language: Source language code
            target_language: Target language code
            
        Returns:
            Translated text
        """
        return self.translator.translate(text, source_language, target_language)
    
    def speech_to_translated_text(
        self, 
        audio: Union[str, bytes, BinaryIO],
        source_language: Optional[str] = None,
        target_language: str = "en"
    ) -> Dict:
        """
        Convert speech to translated text.
        
        Args:
            audio: Audio file path, bytes, or file-like object
            source_language: Source language code (optional)
            target_language: Target language code
            
        Returns:
            Dict containing transcription, translation, and metadata
        """
        # Transcribe audio
        transcription = self.speech_to_text(audio, source_language)
        
        # Determine source language if not provided
        if source_language is None:
            source_language = transcription.get("language", "en")
        
        # Translate text
        translation = self.translate_text(
            transcription["text"], 
            source_language, 
            target_language
        )
        
        return {
            "transcription": transcription["text"],
            "source_language": source_language,
            "translation": translation,
            "target_language": target_language
            }
    
    def speech_to_translated_speech(
        self, 
        audio: Union[str, bytes, BinaryIO],
        source_language: Optional[str] = None,
        target_language: str = "en",
        output_path: Optional[str] = None
    ) -> Tuple[Dict, Union[str, bytes]]:
        """
        Convert speech in one language to speech in another language.
        
        Args:
            audio: Audio file path, bytes, or file-like object
            source_language: Source language code (optional)
            target_language: Target language code
            output_path: Path to save output audio file (optional)
            
        Returns:
            Tuple of (translation_info, audio_output)
        """
        # Convert speech to translated text
        translation_info = self.speech_to_translated_text(
            audio, source_language, target_language
        )
        
        # Synthesize translated text
        audio_output = self.text_to_speech(
            translation_info["translation"], 
            target_language,
            output_path
        )
        
        return translation_info, audio_output
    
    def supported_languages(self) -> Dict[str, str]:
        """
        Get supported languages.
        
        Returns:
            Dict of language codes and names
        """
        return LANGUAGES.copy()