# src/web/components.py
import gradio as gr
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

# Use relative imports
from ..pipeline import EchoLangPipeline
from ..utils.language import LanguageCode

logger = logging.getLogger(__name__)

# Helper function to create language dropdowns consistently
def create_language_dropdown(label: str, default_value: str = LanguageCode.ENGLISH) -> gr.Dropdown:
    return gr.Dropdown(
        label=label,
        choices=LanguageCode.get_choices_for_gradio(),
        value=default_value,
    )

# --- Component Creation Functions ---

def create_stt_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Speech-to-Text tab."""
    with gr.Tab("Speech → Text") as tab:
        gr.Markdown("Transcribe uploaded audio using specific FasterWhisper models.")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech (Upload Only)",
                    type="filepath",
                    sources=["upload"]
                )
                src_lang = create_language_dropdown("Source Language (Hint)", LanguageCode.ENGLISH)
                transcribe_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=2):
                text_output = gr.Textbox(
                    label="Transcription",
                    lines=10,
                    placeholder="Transcribed text will appear here..."
                )

        def handle_transcribe(audio_path: Optional[str], lang_hint: str) -> str:
            # (Function body remains the same as Response #47)
            if not audio_path:
                logger.warning("STT requested but no audio provided.")
                gr.Warning("Please upload audio first.")
                return ""
            logger.info(f"Handling transcription request: audio='{audio_path}', lang_hint='{lang_hint}'")
            try:
                result = pipeline.speech_to_text(audio_path, lang_hint)
                transcription = result.get("text", "ERROR: No text found in result.")
                if "ERROR:" in transcription or result.get("error"):
                     err_msg = result.get("error", transcription)
                     logger.error(f"STT failed: {err_msg}")
                     gr.Error(f"Transcription failed: {err_msg}") # Show error popup
                else:
                     logger.info("Transcription successful.")
                return transcription if "ERROR:" not in transcription else ""
            except Exception as e:
                logger.error(f"STT handler failed: {e}", exc_info=True)
                gr.Error(f"An unexpected error occurred: {e}")
                return f"⚠️ Error: {e}"

        transcribe_btn.click(
            fn=handle_transcribe,
            inputs=[audio_input, src_lang],
            outputs=text_output
        )
    return tab

def create_tts_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Text-to-Speech tab (XTTS/MMS)."""
    with gr.Tab("Text → Speech") as tab:
        gr.Markdown("Synthesize speech using Coqui XTTS-v2 (en/hi) or Facebook MMS-TTS (kn).")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    text_input = gr.Textbox(
                        label="Input Text", lines=8, placeholder="Enter text..."
                    )
                    tts_lang = create_language_dropdown("Language", LanguageCode.ENGLISH)
                    # --- REMOVED Speed Slider ---
                    # speed_slider = gr.Slider(...)
                    # --- End Removal ---
                    speaker_audio = gr.Audio(
                            label="Speaker Reference (Optional WAV - XTTS Only)",
                            type="filepath",
                            sources=["upload"],
                            visible=True # Initial visibility managed by change event
                        )
                synthesize_btn = gr.Button("Synthesize", variant="primary")

            with gr.Column(scale=1):
                with gr.Group():
                    audio_output = gr.Audio(
                        label="Synthesized Speech", type="filepath", autoplay=False
                    )
                    status_output = gr.Textbox(label="Status", interactive=False)

        def toggle_speaker_input(lang_code: str) -> gr.Audio:
            # (Function body remains the same as Response #47)
            if lang_code == LanguageCode.KANNADA:
                logger.debug("Kannada selected for TTS, hiding speaker reference input.")
                return gr.Audio(visible=False)
            else:
                logger.debug(f"{lang_code} selected for TTS, showing speaker reference input.")
                return gr.Audio(visible=True)

        tts_lang.change(
            fn=toggle_speaker_input,
            inputs=[tts_lang],
            outputs=[speaker_audio]
        )

        # --- MODIFIED Handler Signature: Removed 'speed' ---
        def handle_synthesize(text: str, lang: str, speaker_ref_path: Optional[str]) -> Tuple[Optional[str], str]:
        # --- End Modification ---
            status_message = ""
            output_audio = None
            if not text or not text.strip():
                 logger.warning("TTS requested but no text provided.")
                 gr.Warning("Please enter text to synthesize.")
                 status_message = "⚠️ Please enter text."
                 return output_audio, status_message

            # --- MODIFIED Log: Removed 'speed' ---
            logger.info(f"Handling synthesis request: lang='{lang}', speaker='{speaker_ref_path}', text='{text[:50]}...'")
            # --- End Modification ---
            try:
                # --- MODIFIED Call: Removed 'speed' argument ---
                result = pipeline.text_to_speech(text, lang, speaker_ref_path) # Speed argument removed
                # --- End Modification ---
                output_audio = result.get("audio_path")
                error = result.get("error")

                if error:
                     logger.error(f"TTS failed: {error}")
                     gr.Error(f"Synthesis failed: {error}")
                     status_message = f"⚠️ Error: {error}"
                elif output_audio:
                     logger.info("Synthesis successful.")
                     status_message = "✅ Synthesis successful."
                else:
                     logger.error("TTS returned no audio path and no error.")
                     gr.Error("Synthesis failed for an unknown reason.")
                     status_message = "⚠️ Error: Unknown synthesis failure."

            except Exception as e:
                logger.error(f"TTS handler failed: {e}", exc_info=True)
                gr.Error(f"An unexpected error occurred: {e}")
                status_message = f"⚠️ Error: {e}"

            return output_audio, status_message

        # --- MODIFIED Click Inputs: Removed 'speed_slider' ---
        synthesize_btn.click(
            fn=handle_synthesize,
            inputs=[text_input, tts_lang, speaker_audio], # speed_slider removed
            outputs=[audio_output, status_output]
        )
        # --- End Modification ---
    return tab

def create_translation_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Text Translation tab (IndicTrans2)."""
    # (No changes needed in this tab's definition - same as Response #47)
    with gr.Tab("Text Translation") as tab:
        gr.Markdown("Translate text using local IndicTrans2 models.")
        with gr.Row():
            with gr.Column(scale=1):
                source_text = gr.Textbox(
                    label="Source Text", lines=8, placeholder="Enter text..."
                )
                src_lang = create_language_dropdown("Source Language", LanguageCode.ENGLISH)

            with gr.Column(scale=1):
                target_text = gr.Textbox(
                    label="Translated Text", lines=8, placeholder="Translation..."
                )
                tgt_lang = create_language_dropdown("Target Language", LanguageCode.HINDI)

        translate_btn = gr.Button("Translate", variant="primary")

        def handle_translate(text: str, src: str, tgt: str) -> str:
            # (Function body remains the same as Response #47)
            if not text or not text.strip():
                logger.warning("Translation requested but no text provided.")
                gr.Warning("Please enter text to translate.")
                return ""
            if src == tgt:
                 logger.warning(f"Translate src==tgt ({src}). Skipping.")
                 gr.Info("Source and target languages are the same.")
                 return text

            logger.info(f"Handling translation request: {src} -> {tgt}, text='{text[:50]}...'")
            try:
                result = pipeline.translate_text(text, src, tgt)
                translation = result.get("translated_text", "")
                error = result.get("error")

                if error:
                     logger.error(f"Translation failed: {error}")
                     gr.Error(f"Translation failed: {error}")
                     return f"⚠️ Error: {error}"
                elif translation is not None: # Check for None explicitly
                     logger.info("Translation successful.")
                     return translation
                else:
                     logger.warning("Translation successful but result is None/Empty string.")
                     return "" # Return empty string if translation is None or empty

            except Exception as e:
                logger.error(f"Translation handler failed: {e}", exc_info=True)
                gr.Error(f"An unexpected error occurred: {e}")
                return f"⚠️ Error: {e}"


        translate_btn.click(
            fn=handle_translate,
            inputs=[source_text, src_lang, tgt_lang],
            outputs=target_text
        )
    return tab

def create_speech_to_translated_text_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Speech to Translated Text tab (Whisper -> IndicTrans2)."""
    # (No changes needed in this tab's definition - same as Response #47)
    with gr.Tab("Speech → Translated Text") as tab:
        gr.Markdown("Transcribe (FasterWhisper), then translate (IndicTrans2).")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech (Upload Only)", type="filepath", sources=["upload"]
                )
                src_lang = create_language_dropdown("Source Language", LanguageCode.ENGLISH)
                tgt_lang = create_language_dropdown("Target Language", LanguageCode.HINDI)
                process_btn = gr.Button("Process Speech to Translated Text", variant="primary")

            with gr.Column(scale=2):
                 original_text = gr.Textbox(
                    label="Original Transcription", lines=5, placeholder="Transcription..."
                 )
                 translated_text = gr.Textbox(
                    label="Translated Text", lines=5, placeholder="Translation..."
                 )
                 status_output = gr.Textbox(label="Status", interactive=False)

        def handle_speech_to_translated_text(audio_path: Optional[str], src: str, tgt: str) -> Tuple[str, str, str]:
             # (Function body remains the same as Response #47)
            status = ""
            original = ""
            translated = ""
            if not audio_path:
                gr.Warning("Please upload audio first.")
                return original, translated, "⚠️ Please provide audio."

            logger.info(f"Handling Speech -> Translated Text: audio='{audio_path}', {src} -> {tgt}")
            try:
                result = pipeline.speech_to_translated_text(audio_path, src, tgt)
                transcription_res = result.get("transcription", {})
                translation_res = result.get("translation", {})

                original = transcription_res.get("text", "")
                translated = translation_res.get("translated_text", "")
                # Check for errors in both stages
                transcription_error_msg = transcription_res.get("error")
                translation_error_msg = translation_res.get("error")
                transcription_text_error = original.startswith("ERROR:") if isinstance(original, str) else False
                translation_text_error = translated.startswith("ERROR:") if isinstance(translated, str) else False

                final_error = transcription_error_msg or (original if transcription_text_error else None) \
                              or translation_error_msg or (translated if translation_text_error else None)

                if final_error:
                    status = f"⚠️ Error during processing: {final_error}"
                    logger.error(status)
                    gr.Error(status)
                    if transcription_error_msg or transcription_text_error: original = ""
                    if translation_error_msg or translation_text_error: translated = ""
                elif src == tgt and original:
                    status = "✅ Transcription complete (translation not needed)."
                    translated = original
                elif original and translated is not None:
                     status = "✅ Processing complete."
                     logger.info(status)
                else:
                     status = "⚠️ Processing failed for an unknown reason (check logs)."
                     logger.error(status + f" Results: {result}")
                     gr.Error(status)

            except Exception as e:
                logger.error(f"Speech -> Translated Text handler failed: {e}", exc_info=True)
                status = f"⚠️ Unexpected Error: {e}"
                gr.Error(status)
                original = ""
                translated = ""

            return original, translated, status

        process_btn.click(
            fn=handle_speech_to_translated_text,
            inputs=[audio_input, src_lang, tgt_lang],
            outputs=[original_text, translated_text, status_output]
        )
    return tab


def create_speech_to_translated_speech_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Speech to Translated Speech tab (Whisper -> IndicTrans2 -> XTTS/MMS)."""
    with gr.Tab("Speech → Translated Speech") as tab:
        gr.Markdown("Transcribe (FasterWhisper), translate (IndicTrans2), synthesize (XTTS/MMS).")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech (Upload Only)", type="filepath", sources=["upload"]
                )
                src_lang = create_language_dropdown("Source Language", LanguageCode.ENGLISH)
                tgt_lang = create_language_dropdown("Target Language", LanguageCode.HINDI)
                # --- REMOVED Speed Slider ---
                # speed_slider = gr.Slider(...)
                # --- End Removal ---
                speaker_audio = gr.Audio(
                    label="Output Speaker Ref (Optional WAV - XTTS Only)", type="filepath", sources=["upload"], visible=(tgt_lang != LanguageCode.KANNADA)
                )
                process_btn = gr.Button("Process Speech to Translated Speech", variant="primary")

            with gr.Column(scale=2):
                 original_text = gr.Textbox(
                    label="Original Transcription", lines=4, placeholder="..."
                 )
                 translated_text = gr.Textbox(
                    label="Translated Text", lines=4, placeholder="..."
                 )
                 audio_output = gr.Audio(
                    label="Translated Speech", type="filepath", autoplay=False
                 )
                 status_output = gr.Textbox(label="Status", interactive=False)

        def toggle_speaker_input_s2s(target_lang_code: str) -> gr.Audio:
             # (Function body remains the same as Response #47)
            if target_lang_code == LanguageCode.KANNADA:
                logger.debug("Target is Kannada for S2S TTS, hiding speaker reference input.")
                return gr.Audio(visible=False)
            else:
                logger.debug(f"Target is {target_lang_code} for S2S TTS, showing speaker reference input.")
                return gr.Audio(visible=True)


        tgt_lang.change(
            fn=toggle_speaker_input_s2s,
            inputs=[tgt_lang],
            outputs=[speaker_audio]
        )

        # --- MODIFIED Handler Signature: Removed 'speed' ---
        def handle_speech_to_translated_speech(audio_path: Optional[str], src: str, tgt: str, speaker_ref_path: Optional[str]) -> Tuple[str, str, Optional[str], str]:
        # --- End Modification ---
            status = ""
            original = ""
            translated = ""
            output_audio = None

            if not audio_path:
                gr.Warning("Please upload audio first.")
                return original, translated, output_audio, "⚠️ Please provide audio."

            if tgt == LanguageCode.KANNADA:
                speaker_ref_path = None # Ensure speaker ref is None if target is Kannada

            # --- MODIFIED Log: Removed 'speed' ---
            logger.info(f"Handling Speech -> Translated Speech: audio='{audio_path}', {src} -> {tgt}, speaker='{speaker_ref_path}'")
            # --- End Modification ---
            try:
                 # --- MODIFIED Call: Removed 'speed' argument ---
                result = pipeline.speech_to_translated_speech(audio_path, src, tgt, speaker_ref_path) # Speed argument removed
                 # --- End Modification ---
                transcription_res = result.get("transcription", {})
                translation_res = result.get("translation", {})
                synthesis_res = result.get("synthesis", {})

                original = transcription_res.get("text", "")
                translated = translation_res.get("translated_text", "")
                output_audio = synthesis_res.get("audio_path")

                # Consolidate errors from all stages
                transcription_error_msg = transcription_res.get("error")
                translation_error_msg = translation_res.get("error")
                synthesis_error_msg = synthesis_res.get("error")
                transcription_text_error = original.startswith("ERROR:") if isinstance(original, str) else False
                translation_text_error = translated.startswith("ERROR:") if isinstance(translated, str) else False

                final_error = transcription_error_msg or (original if transcription_text_error else None) \
                              or translation_error_msg or (translated if translation_text_error else None) \
                              or synthesis_error_msg

                if final_error:
                    status = f"⚠️ Error during processing: {final_error}"
                    logger.error(status)
                    gr.Error(status)
                    if transcription_error_msg or transcription_text_error: original = ""
                    if translation_error_msg or translation_text_error: translated = ""
                    if synthesis_error_msg: output_audio = None
                elif src == tgt and original and output_audio:
                    status = "✅ Transcription and resynthesis complete."
                    translated = original
                elif original and translated is not None and output_audio:
                     status = "✅ Processing complete."
                else:
                     status = "⚠️ Processing failed for an unknown reason (check logs)."
                     logger.error(status + f" Results: {result}")
                     gr.Error(status)

            except Exception as e:
                logger.error(f"Speech -> Translated Speech handler failed: {e}", exc_info=True)
                status = f"⚠️ Unexpected Error: {e}"
                gr.Error(status)
                original = ""
                translated = ""
                output_audio = None

            return original, translated, output_audio, status

        # --- MODIFIED Click Inputs: Removed 'speed_slider' ---
        process_btn.click(
            fn=handle_speech_to_translated_speech,
            inputs=[audio_input, src_lang, tgt_lang, speaker_audio], # speed_slider removed
            outputs=[original_text, translated_text, audio_output, status_output]
        )
        # --- End Modification ---
    return tab