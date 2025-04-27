# src/pipeline.py
import os
import tempfile
from typing import Dict, Optional, Union
from pathlib import Path

from .stt.stt import SpeechToText
from .translation.translator import Translator
from .tts.synthesizer import TextToSpeech
from .utils.model_utils import ModelManager
from .utils.language import LanguageCode


class EchoLangPipeline:
    """
    End-to-end pipeline for EchoLang.
    """
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize the EchoLang pipeline."""
        self.model_manager = model_manager or ModelManager()
        self.stt = SpeechToText(self.model_manager)
        self.translator = Translator(self.model_manager)
        self.tts = TextToSpeech(self.model_manager)
        
    def speech_to_text(self, 
                     audio_path: Union[str, Path],
                     src_lang: str = LanguageCode.ENGLISH) -> Dict:
        """
        Convert speech to text.
        
        Args:
            audio_path: Path to audio file
            src_lang: Source language code
            
        Returns:
            Dictionary with transcription results
        """
        return self.stt.transcribe(audio_path, src_lang)
        
    def text_to_speech(self,
                      text: str,
                      lang: str = LanguageCode.ENGLISH,
                      speaker_audio: Optional[str] = None,
                      speed: float = 1.0) -> Dict:
        """
        Convert text to speech.
        
        Args:
            text: Text to synthesize
            lang: Language code
            speaker_audio: Path to speaker reference audio (optional)
            speed: Speech speed factor
            
        Returns:
            Dictionary with synthesis results
        """
        return self.tts.synthesize(text, lang, speaker_audio, speed)
        
    def translate_text(self,
                     text: str,
                     src_lang: str = LanguageCode.ENGLISH,
                     tgt_lang: str = LanguageCode.HINDI) -> Dict:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            Dictionary with translation results
        """
        return self.translator.translate(text, src_lang, tgt_lang)
        
    def speech_to_translated_text(self,
                                audio_path: Union[str, Path],
                                src_lang: str = LanguageCode.ENGLISH,
                                tgt_lang: str = LanguageCode.HINDI) -> Dict:
        """
        Convert speech to translated text.
        
        Args:
            audio_path: Path to audio file
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            Dictionary with transcription and translation results
        """
        # First transcribe the speech
        transcription_result = self.speech_to_text(audio_path, src_lang)
        
        # Then translate the transcription
        translation_result = self.translate_text(
            text=transcription_result["text"],
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        
        return {
            "transcription": transcription_result,
            "translation": translation_result
        }
        
    def speech_to_translated_speech(self,
                                  audio_path: Union[str, Path],
                                  src_lang: str = LanguageCode.ENGLISH,
                                  tgt_lang: str = LanguageCode.HINDI,
                                  speaker_audio: Optional[str] = None,
                                  speed: float = 1.0) -> Dict:
        """
        Convert speech to translated speech.
        
        Args:
            audio_path: Path to audio file
            src_lang: Source language code
            tgt_lang: Target language code
            speaker_audio: Path to speaker reference audio (optional)
            speed: Speech speed factor
            
        Returns:
            Dictionary with complete pipeline results
        """
        # First convert speech to translated text
        speech_to_text_result = self.speech_to_translated_text(audio_path, src_lang, tgt_lang)
        
        # Then synthesize the translated text
        translated_text = speech_to_text_result["translation"]["translated_text"]
        
        synthesis_result = self.text_to_speech(
            text=translated_text,
            lang=tgt_lang,
            speaker_audio=speaker_audio,
            speed=speed
        )
        
        return {
            "transcription": speech_to_text_result["transcription"],
            "translation": speech_to_text_result["translation"],
            "synthesis": synthesis_result
        }