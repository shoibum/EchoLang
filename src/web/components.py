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
        choices=LanguageCode.get_choices_for_gradio(), # Use helper from LanguageCode
        value=default_value,
        # Allow None for Whisper auto-detect? Maybe add an "Auto-Detect" option.
        # For now, require selection.
        # allow_custom_value=False # Ensure only valid codes are selected
    )

# --- Component Creation Functions ---

def create_stt_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Speech-to-Text tab (Whisper)."""
    with gr.Tab("Speech → Text") as tab:
        gr.Markdown("Transcribe audio using OpenAI Whisper.")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech",
                    type="filepath",
                    sources=["microphone", "upload"]
                )
                # Whisper can auto-detect, but hinting improves accuracy/speed
                src_lang = create_language_dropdown("Source Language (Hint)", LanguageCode.ENGLISH)
                transcribe_btn = gr.Button("Transcribe", variant="primary")

            with gr.Column(scale=2):
                text_output = gr.Textbox(
                    label="Transcription",
                    lines=10,
                    placeholder="Transcribed text will appear here..."
                )
                # Optional: Display detected language if Whisper provides it easily
                # lang_output = gr.Textbox(label="Detected Language")

        def handle_transcribe(audio_path: Optional[str], lang_hint: str) -> str:
            if not audio_path:
                logger.warning("STT requested but no audio provided.")
                gr.Warning("Please record or upload audio first.")
                return "" # Return empty string for Textbox
            logger.info(f"Handling transcription request: audio='{audio_path}', lang_hint='{lang_hint}'")
            try:
                # Pass the language hint to the pipeline
                result = pipeline.speech_to_text(audio_path, lang_hint)
                transcription = result.get("text", "ERROR: No text found in result.")
                if "ERROR:" in transcription:
                     logger.error(f"STT failed: {transcription}")
                     gr.Error(f"Transcription failed: {transcription}") # Show error popup
                else:
                     logger.info("Transcription successful.")
                return transcription
            except Exception as e:
                logger.error(f"STT handler failed: {e}", exc_info=True)
                gr.Error(f"An unexpected error occurred: {e}")
                return f"⚠️ Error: {e}"

        transcribe_btn.click(
            fn=handle_transcribe,
            inputs=[audio_input, src_lang],
            outputs=text_output # Add lang_output if implemented
        )
    return tab

def create_tts_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Text-to-Speech tab (XTTS)."""
    with gr.Tab("Text → Speech") as tab:
        gr.Markdown("Synthesize speech using Coqui XTTS-v2.")
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Input Text", lines=8, placeholder="Enter text..."
                )
                tts_lang = create_language_dropdown("Language", LanguageCode.ENGLISH)
                speed_slider = gr.Slider(
                    label="Speed", minimum=0.5, maximum=1.5, value=1.0, step=0.1
                )
                speaker_audio = gr.Audio(
                        label="Speaker Reference (Optional WAV)",
                        # info="Upload a short WAV file (~5-15s). If blank, uses default voice.",
                        type="filepath",
                        sources=["upload"]
                        # Removed: accept_multiple_files=False
                    )
                synthesize_btn = gr.Button("Synthesize", variant="primary")

            with gr.Column(scale=1):
                audio_output = gr.Audio(
                    label="Synthesized Speech", type="filepath", autoplay=False
                )
                # Add status textbox for errors
                status_output = gr.Textbox(label="Status", interactive=False)

        def handle_synthesize(text: str, lang: str, speed: float, speaker_ref_path: Optional[str]) -> Tuple[Optional[str], str]:
            status_message = ""
            output_audio = None
            if not text or not text.strip():
                 logger.warning("TTS requested but no text provided.")
                 gr.Warning("Please enter text to synthesize.")
                 status_message = "⚠️ Please enter text."
                 return output_audio, status_message

            logger.info(f"Handling synthesis request: lang='{lang}', speed='{speed}', speaker='{speaker_ref_path}', text='{text[:50]}...'")
            try:
                result = pipeline.text_to_speech(text, lang, speaker_ref_path, speed)
                output_audio = result.get("audio_path") # Path or None
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

        synthesize_btn.click(
            fn=handle_synthesize,
            inputs=[text_input, tts_lang, speed_slider, speaker_audio],
            outputs=[audio_output, status_output] # Output to Audio and Status Textbox
        )
    return tab

def create_translation_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Text Translation tab (NLLB)."""
    with gr.Tab("Text Translation") as tab:
        gr.Markdown("Translate text using Meta NLLB.")
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
            if not text or not text.strip():
                logger.warning("Translation requested but no text provided.")
                gr.Warning("Please enter text to translate.")
                return ""
            if src == tgt:
                 logger.warning(f"Translate src==tgt ({src}). Skipping.")
                 gr.Info("Source and target languages are the same.")
                 return text # Return original

            logger.info(f"Handling translation request: {src} -> {tgt}, text='{text[:50]}...'")
            try:
                result = pipeline.translate_text(text, src, tgt)
                translation = result.get("translated_text", "")
                error = result.get("error")

                if error:
                     logger.error(f"Translation failed: {error}")
                     gr.Error(f"Translation failed: {error}")
                     return f"⚠️ Error: {error}" # Display error in output box
                elif translation:
                     logger.info("Translation successful.")
                     return translation
                else:
                     logger.error("Translation returned no text and no error.")
                     gr.Error("Translation failed for an unknown reason.")
                     return "⚠️ Error: Unknown translation failure."

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
    """Create Speech to Translated Text tab (Whisper -> NLLB)."""
    with gr.Tab("Speech → Translated Text") as tab:
        gr.Markdown("Transcribe with Whisper, then translate with NLLB.")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech", type="filepath", sources=["microphone", "upload"]
                )
                src_lang = create_language_dropdown("Source Language", LanguageCode.ENGLISH)
                tgt_lang = create_language_dropdown("Target Language", LanguageCode.HINDI)

            with gr.Column(scale=2):
                 original_text = gr.Textbox(
                    label="Original Transcription (Whisper)", lines=5, placeholder="Transcription..."
                 )
                 translated_text = gr.Textbox(
                    label="Translated Text (NLLB)", lines=5, placeholder="Translation..."
                 )
                 status_output = gr.Textbox(label="Status", interactive=False)


        process_btn = gr.Button("Process Speech to Translated Text", variant="primary")

        def handle_speech_to_translated_text(audio_path: Optional[str], src: str, tgt: str) -> Tuple[str, str, str]:
            status = ""
            original = ""
            translated = ""
            if not audio_path:
                gr.Warning("Please record or upload audio first.")
                return original, translated, "⚠️ Please provide audio."
            if src == tgt:
                gr.Info("Source and target languages are the same. Only transcription will be performed.")
                # Still run pipeline, but translation won't change text
                # Fall through...

            logger.info(f"Handling Speech -> Translated Text: audio='{audio_path}', {src} -> {tgt}")
            try:
                result = pipeline.speech_to_translated_text(audio_path, src, tgt)
                transcription_res = result.get("transcription", {})
                translation_res = result.get("translation", {})

                original = transcription_res.get("text", "")
                translated = translation_res.get("translated_text", "")
                transcription_error = original.startswith("ERROR:") if original else False
                translation_error = translation_res.get("error")

                if transcription_error:
                    status = f"⚠️ Transcription failed: {original}"
                    logger.error(status)
                    gr.Error(status)
                    original = "" # Clear output box on error
                    translated = ""
                elif translation_error:
                     status = f"⚠️ Translation failed: {translation_error}"
                     logger.error(status)
                     gr.Error(status)
                     translated = "" # Clear translation box
                elif src == tgt and original: # Handle src==tgt case after transcription
                    status = "✅ Transcription complete (translation not needed)."
                    translated = original # Show original in translation box
                elif original and translated:
                     status = "✅ Processing complete."
                     logger.info(status)
                else:
                     status = "⚠️ Processing failed for an unknown reason."
                     logger.error(status)
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
    """Create Speech to Translated Speech tab (Whisper -> NLLB -> XTTS)."""
    with gr.Tab("Speech → Translated Speech") as tab:
        gr.Markdown("Transcribe (Whisper), translate (NLLB), synthesize (XTTS).")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech", type="filepath", sources=["microphone", "upload"]
                )
                src_lang = create_language_dropdown("Source Language", LanguageCode.ENGLISH)
                tgt_lang = create_language_dropdown("Target Language", LanguageCode.HINDI)
                speed_slider = gr.Slider(
                    label="Output Speech Speed", minimum=0.5, maximum=1.5, value=1.0, step=0.1
                )
                speaker_audio = gr.Audio(
                    label="Output Speaker Ref (Optional WAV)", type="filepath", sources=["upload"]
                )

            with gr.Column(scale=2):
                 original_text = gr.Textbox(
                    label="Original Transcription (Whisper)", lines=4, placeholder="..."
                 )
                 translated_text = gr.Textbox(
                    label="Translated Text (NLLB)", lines=4, placeholder="..."
                 )
                 audio_output = gr.Audio(
                    label="Translated Speech (XTTS)", type="filepath", autoplay=False
                 )
                 status_output = gr.Textbox(label="Status", interactive=False)

        process_btn = gr.Button("Process Speech to Translated Speech", variant="primary")

        def handle_speech_to_translated_speech(audio_path: Optional[str], src: str, tgt: str, speed: float, speaker_ref_path: Optional[str]) -> Tuple[str, str, Optional[str], str]:
            status = ""
            original = ""
            translated = ""
            output_audio = None

            if not audio_path:
                gr.Warning("Please record or upload audio first.")
                return original, translated, output_audio, "⚠️ Please provide audio."

            logger.info(f"Handling Speech -> Translated Speech: audio='{audio_path}', {src} -> {tgt}, speed={speed}, speaker='{speaker_ref_path}'")
            try:
                result = pipeline.speech_to_translated_speech(audio_path, src, tgt, speaker_ref_path, speed)
                transcription_res = result.get("transcription", {})
                translation_res = result.get("translation", {})
                synthesis_res = result.get("synthesis", {})

                original = transcription_res.get("text", "")
                translated = translation_res.get("translated_text", "")
                output_audio = synthesis_res.get("audio_path")
                transcription_error = original.startswith("ERROR:") if original else False
                translation_error = translation_res.get("error")
                synthesis_error = synthesis_res.get("error")

                if transcription_error:
                    status = f"⚠️ Transcription failed: {original}"
                    original = "" # Clear boxes
                    translated = ""
                elif translation_error:
                    status = f"⚠️ Translation failed: {translation_error}"
                    translated = "" # Clear boxes
                elif synthesis_error:
                     status = f"⚠️ Synthesis failed: {synthesis_error}"
                     # Keep text boxes filled, but clear audio
                     output_audio = None
                elif src == tgt and original and output_audio: # Handle src==tgt case if synthesis worked
                    status = "✅ Transcription and resynthesis complete."
                    translated = original # Show original in translation box
                elif original and translated and output_audio:
                     status = "✅ Processing complete."
                else:
                     status = "⚠️ Processing failed for an unknown reason."

                if status.startswith("⚠️"):
                     logger.error(status)
                     gr.Error(status) # Show error popup
                else:
                     logger.info(status)

            except Exception as e:
                logger.error(f"Speech -> Translated Speech handler failed: {e}", exc_info=True)
                status = f"⚠️ Unexpected Error: {e}"
                gr.Error(status)
                original = ""
                translated = ""
                output_audio = None

            return original, translated, output_audio, status

        process_btn.click(
            fn=handle_speech_to_translated_speech,
            inputs=[audio_input, src_lang, tgt_lang, speed_slider, speaker_audio],
            outputs=[original_text, translated_text, audio_output, status_output]
        )
    return tab