# src/pipeline.py
import os
import tempfile
from typing import Dict, Optional, Union
from pathlib import Path
import logging

# Updated imports for model wrappers
from .stt.stt import SpeechToText
from .translation.translator import Translator
from .tts.synthesizer import TextToSpeech
from .utils.model_utils import ModelManager
from .utils.language import LanguageCode

logger = logging.getLogger(__name__)

class EchoLangPipeline:
    """
    End-to-end pipeline for EchoLang (FasterWhisper + MarianMT/NLLB + XTTS).
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        """Initialize the EchoLang pipeline."""
        logger.info("Initializing EchoLangPipeline...")
        # Ensure model_manager is an instance of ModelManager, create if None
        self.model_manager = model_manager if isinstance(model_manager, ModelManager) else ModelManager()

        try:
            # Initialize components, passing the confirmed ModelManager instance
            logger.info("Initializing SpeechToText (FasterWhisper) component...")
            self.stt = SpeechToText(self.model_manager)

            logger.info("Initializing Translator (MarianMT/NLLB) component...")
            self.translator = Translator(self.model_manager)

            logger.info("Initializing TextToSpeech (XTTS) component...")
            self.tts = TextToSpeech(self.model_manager)

            logger.info("EchoLangPipeline components initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}", exc_info=True)
            # Propagate the error clearly
            raise RuntimeError(f"Pipeline component initialization failed: {e}") from e

    # --- STT Method ---
    def speech_to_text(self,
                       audio_path: Union[str, Path],
                       src_lang: Optional[str] = None) -> Dict:
        """
        Convert speech to text using FasterWhisper.

        Args:
            audio_path: Path to audio file
            src_lang: Source language code hint (optional)

        Returns:
            Dictionary with transcription results {'text': str, 'language': str}
        """
        logger.info(f"Performing STT for audio: {audio_path}, language hint: {src_lang}")
        try:
            result = self.stt.transcribe(audio_path, src_lang)
            logger.debug(f"STT result: {result}")
            return result
        except Exception as e:
            logger.error(f"STT pipeline step failed for {audio_path}: {e}", exc_info=True)
            return {"text": f"ERROR: STT failed - {type(e).__name__}", "language": src_lang or "unknown"}

    # --- TTS Method ---
    def text_to_speech(self,
                       text: str,
                       lang: str = LanguageCode.ENGLISH,
                       speaker_audio: Optional[str] = None,
                       speed: float = 1.0) -> Dict:
        """
        Convert text to speech using XTTS (via TTS API wrapper on CPU).

        Args:
            text: Text to synthesize
            lang: Language code
            speaker_audio: Path to speaker reference audio (optional)
            speed: Speech speed factor

        Returns:
            Dictionary with synthesis results {'audio_path': str|None, ..., 'error': str|None}
        """
        logger.info(f"Performing TTS for text: '{text[:50]}...', language: {lang}, speed: {speed}")
        if speaker_audio: logger.info(f"Using speaker reference audio: {speaker_audio}")
        try:
            result = self.tts.synthesize(text, lang, speaker_audio, speed)
            logger.debug(f"TTS result: {result}")
            return result
        except Exception as e:
            logger.error(f"TTS pipeline step failed for text '{text[:50]}...': {e}", exc_info=True)
            return {"audio_path": None, "text": text, "language": lang, "error": f"TTS failed - {type(e).__name__}"}

    # --- Translation Method ---
    def translate_text(self,
                       text: str,
                       src_lang: str = LanguageCode.ENGLISH,
                       tgt_lang: str = LanguageCode.HINDI) -> Dict:
        """
        Translate text using MarianMT or NLLB fallback.

        Args:
            text: Text to translate
            src_lang: Source language code (internal)
            tgt_lang: Target language code (internal)

        Returns:
            Dictionary with translation results {'original_text': str, 'translated_text': str, ..., 'error': str|None}
        """
        logger.info(f"Performing translation from {src_lang} to {tgt_lang} for text: '{text[:50]}...'")
        try:
            result = self.translator.translate(text, src_lang, tgt_lang)
            logger.debug(f"Translation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Translation pipeline step failed for text '{text[:50]}...': {e}", exc_info=True)
            return {"original_text": text, "translated_text": "", "src_lang": src_lang, "tgt_lang": tgt_lang, "error": f"Translation failed - {type(e).__name__}"}

    # --- Combined Methods ---
    def speech_to_translated_text(self,
                                  audio_path: Union[str, Path],
                                  src_lang: str = LanguageCode.ENGLISH,
                                  tgt_lang: str = LanguageCode.HINDI) -> Dict:
        """Convert speech to translated text (FasterWhisper -> MarianMT/NLLB)."""
        logger.info(f"Performing Speech->Translated Text: {audio_path}, {src_lang} -> {tgt_lang}")

        # Step 1: Transcribe
        transcription_result = self.speech_to_text(audio_path, src_lang)
        original_text = transcription_result.get("text", "")
        transcription_error = transcription_result.get("error")

        # Handle transcription errors
        if transcription_error or not original_text:
             logger.warning(f"Transcription step failed or yielded no text: {transcription_error or 'Empty text'}")
             # Ensure consistent error structure
             if not transcription_error: transcription_error = "Transcription failed (empty result)"
             return {
                 "transcription": transcription_result,
                 "translation": {"original_text": original_text, "translated_text": "", "src_lang": src_lang, "tgt_lang": tgt_lang, "error": transcription_error}
             }

        # Step 2: Translate
        translation_result = self.translate_text(
            text=original_text,
            src_lang=src_lang, # Use the user-selected source language for translation intent
            tgt_lang=tgt_lang
        )

        final_result = {
            "transcription": transcription_result,
            "translation": translation_result
        }
        logger.debug(f"Speech->Translated Text result: {final_result}")
        return final_result

    def speech_to_translated_speech(self,
                                    audio_path: Union[str, Path],
                                    src_lang: str = LanguageCode.ENGLISH,
                                    tgt_lang: str = LanguageCode.HINDI,
                                    speaker_audio: Optional[str] = None,
                                    speed: float = 1.0) -> Dict:
        """Convert speech to translated speech (FasterWhisper -> MarianMT/NLLB -> XTTS)."""
        logger.info(f"Performing Speech->Translated Speech: {audio_path}, {src_lang} -> {tgt_lang}, speed: {speed}")
        if speaker_audio: logger.info(f"Using speaker reference audio: {speaker_audio}")

        # Step 1 & 2: Get translated text
        speech_to_text_result = self.speech_to_translated_text(audio_path, src_lang, tgt_lang)

        transcription_result = speech_to_text_result.get("transcription", {})
        translation_result = speech_to_text_result.get("translation", {})
        translated_text = translation_result.get("translated_text", "")
        prior_error = transcription_result.get("error") or translation_result.get("error")

        # Handle errors from previous steps
        if prior_error or not translated_text:
            logger.warning(f"Transcription/Translation step failed or yielded no text for synthesis: {prior_error or 'Empty text'}")
            # Ensure consistent error structure
            if not prior_error: prior_error = "Synthesis failed due to prior step error (empty text)"
            return {
                 "transcription": transcription_result,
                 "translation": translation_result,
                 "synthesis": {"audio_path": None, "text": "", "language": tgt_lang, "error": prior_error}
             }

        # Step 3: Synthesize translated text
        synthesis_result = self.text_to_speech(
            text=translated_text,
            lang=tgt_lang, # Synthesize in the target language
            speaker_audio=speaker_audio,
            speed=speed
        )

        final_result = {
            "transcription": transcription_result,
            "translation": translation_result,
            "synthesis": synthesis_result
        }
        logger.debug(f"Speech->Translated Speech result: {final_result}")
        return final_result