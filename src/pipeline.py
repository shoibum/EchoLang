# src/pipeline.py
import os
import tempfile
from typing import Dict, Optional, Union
from pathlib import Path
import logging

# Updated imports for model wrappers
from .stt.stt import SpeechToText
from .translation.translator import Translator # Using local IndicTrans2
from .tts.synthesizer import TextToSpeech
from .utils.model_utils import ModelManager
from .utils.language import LanguageCode

logger = logging.getLogger(__name__)

class EchoLangPipeline:
    """
    End-to-end pipeline for EchoLang (FasterWhisper + IndicTrans2 + MMS/XTTS).
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        # (Keep __init__ method exactly as in Response #47)
        logger.info("Initializing EchoLangPipeline...")
        self.model_manager = model_manager if isinstance(model_manager, ModelManager) else ModelManager()
        try:
            logger.info("Initializing SpeechToText (FasterWhisper) component...")
            self.stt = SpeechToText(self.model_manager)
            logger.info("Initializing Translator (IndicTrans2 Local) component...")
            self.translator = Translator()
            logger.info("Initializing TextToSpeech (XTTS/MMS) component...")
            self.tts = TextToSpeech(self.model_manager)
            logger.info("EchoLangPipeline components initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline component initialization failed: {e}") from e

    # --- Keep STT Method ---
    def speech_to_text(self,
                       audio_path: Union[str, Path],
                       src_lang: Optional[str] = None) -> Dict:
        # (No changes needed)
        logger.info(f"Performing STT for audio: {audio_path}, language hint: {src_lang}")
        try:
            result = self.stt.transcribe(audio_path, src_lang)
            logger.debug(f"STT result: {result}")
            return result
        except Exception as e:
            logger.error(f"STT pipeline step failed for {audio_path}: {e}", exc_info=True)
            return {"text": f"ERROR: STT failed - {type(e).__name__}", "language": src_lang or "unknown"}


    # --- MODIFIED TTS Method Signature: Removed 'speed' ---
    def text_to_speech(self,
                       text: str,
                       lang: str = LanguageCode.ENGLISH,
                       speaker_audio: Optional[str] = None) -> Dict: # Removed speed=1.0 default
    # --- End Modification ---
        """
        Convert text to speech using the appropriate TTS model (MMS/XTTS).
        """
        # --- MODIFIED Log: Removed 'speed' ---
        logger.info(f"Performing TTS for text: '{text[:50]}...', language: {lang}")
        # --- End Modification ---
        if speaker_audio: logger.info(f"Using speaker reference audio: {speaker_audio}")
        try:
            # --- MODIFIED Call: Removed 'speed' argument ---
            result = self.tts.synthesize(text, lang, speaker_audio) # Speed argument removed
            # --- End Modification ---
            logger.debug(f"TTS result: {result}")
            return result
        except Exception as e:
            logger.error(f"TTS pipeline step failed for text '{text[:50]}...': {e}", exc_info=True)
            # Return error dict without speed
            return {"audio_path": None, "text": text, "language": lang, "error": f"TTS failed - {type(e).__name__}"}

    # --- Keep Translation Method ---
    def translate_text(self,
                       text: str,
                       src_lang: str = LanguageCode.ENGLISH,
                       tgt_lang: str = LanguageCode.HINDI) -> Dict:
        # (No changes needed)
        logger.info(f"Performing local translation from {src_lang} to {tgt_lang} for text: '{text[:50]}...'")
        try:
            result = self.translator.translate(text, src_lang, tgt_lang)
            logger.debug(f"Translation result: {result}")
            return result
        except Exception as e:
            logger.error(f"Translation pipeline step failed for text '{text[:50]}...': {e}", exc_info=True)
            return {"original_text": text, "translated_text": "", "src_lang": src_lang, "tgt_lang": tgt_lang, "error": f"Translation failed - {type(e).__name__}"}

    # --- Keep Speech -> Translated Text Method ---
    def speech_to_translated_text(self,
                                  audio_path: Union[str, Path],
                                  src_lang: str = LanguageCode.ENGLISH,
                                  tgt_lang: str = LanguageCode.HINDI) -> Dict:
        # (No changes needed)
        logger.info(f"Performing Speech->Translated Text: {audio_path}, {src_lang} -> {tgt_lang}")
        transcription_result = self.speech_to_text(audio_path, src_lang)
        original_text = transcription_result.get("text", "")
        transcription_error = transcription_result.get("error")

        if transcription_error or not original_text or original_text.startswith("ERROR:"):
             logger.warning(f"Transcription step failed or yielded no text: {transcription_error or original_text or 'Empty text'}")
             if not transcription_error: transcription_error = "Transcription failed (empty result)"
             return {
                 "transcription": transcription_result,
                 "translation": {"original_text": original_text, "translated_text": "", "src_lang": src_lang, "tgt_lang": tgt_lang, "error": transcription_error}
             }

        stt_detected_lang = transcription_result.get("language", src_lang)
        valid_src_lang = stt_detected_lang if stt_detected_lang in [LanguageCode.ENGLISH, LanguageCode.HINDI, LanguageCode.KANNADA] else src_lang

        translation_result = self.translate_text(
            text=original_text,
            src_lang=valid_src_lang,
            tgt_lang=tgt_lang
        )
        final_result = { "transcription": transcription_result, "translation": translation_result }
        logger.debug(f"Speech->Translated Text result: {final_result}")
        return final_result

    # --- MODIFIED Speech -> Translated Speech Method Signature: Removed 'speed' ---
    def speech_to_translated_speech(self,
                                    audio_path: Union[str, Path],
                                    src_lang: str = LanguageCode.ENGLISH,
                                    tgt_lang: str = LanguageCode.HINDI,
                                    speaker_audio: Optional[str] = None) -> Dict: # Removed speed=1.0 default
    # --- End Modification ---
        """Convert speech to translated speech (FasterWhisper -> IndicTrans2 -> MMS/XTTS)."""
        # --- MODIFIED Log: Removed 'speed' ---
        logger.info(f"Performing Speech->Translated Speech: {audio_path}, {src_lang} -> {tgt_lang}")
        # --- End Modification ---
        if speaker_audio: logger.info(f"Using speaker reference audio: {speaker_audio}")

        # Step 1 & 2: Get translated text
        speech_to_text_result = self.speech_to_translated_text(audio_path, src_lang, tgt_lang)

        transcription_result = speech_to_text_result.get("transcription", {})
        translation_result = speech_to_text_result.get("translation", {})
        translated_text = translation_result.get("translated_text", "")
        # Consolidate prior errors
        prior_error = transcription_result.get("error") or translation_result.get("error")
        if not prior_error and transcription_result.get("text","").startswith("ERROR:"):
             prior_error = transcription_result.get("text")
        if not prior_error and translation_result.get("translated_text","").startswith("ERROR:"):
             prior_error = translation_result.get("translated_text")

        if prior_error or not translated_text:
            logger.warning(f"Transcription/Translation step failed or yielded no text for synthesis: {prior_error or 'Empty text'}")
            if not prior_error: prior_error = "Synthesis failed due to prior step error (empty text)"
            return {
                 "transcription": transcription_result,
                 "translation": translation_result,
                 "synthesis": {"audio_path": None, "text": translated_text, "language": tgt_lang, "error": prior_error}
             }

        # Step 3: Synthesize translated text
        # --- MODIFIED Call: Removed 'speed' argument ---
        synthesis_result = self.text_to_speech(
            text=translated_text,
            lang=tgt_lang,
            speaker_audio=speaker_audio
            # speed argument removed
        )
        # --- End Modification ---

        final_result = {
            "transcription": transcription_result,
            "translation": translation_result,
            "synthesis": synthesis_result
        }
        logger.debug(f"Speech->Translated Speech result: {final_result}")
        return final_result