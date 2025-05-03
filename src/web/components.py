# src/web/components.py (Upload Only Version - Confirmed)
import gradio as gr
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import numpy as np
import os
import time

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

# --- NO save_numpy_audio_to_temp_wav HELPER NEEDED ---

# --- Component Creation Functions ---

def create_stt_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Speech-to-Text tab."""
    with gr.Tab("Speech → Text") as tab:
        gr.Markdown("Transcribe **uploaded** audio using specific FasterWhisper models.")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech (Upload File)",
                    type="filepath", # Expect filepath from upload
                    sources=["upload"] # ONLY Upload
                )
                src_lang = create_language_dropdown("Source Language (Hint)", LanguageCode.ENGLISH)
                transcribe_btn = gr.Button("Transcribe", variant="primary")
            with gr.Column(scale=2):
                text_output = gr.Textbox( label="Transcription", lines=10, placeholder="Transcribed text will appear here...")

        def handle_transcribe(audio_path: Optional[str], lang_hint: str) -> str:
            logger.info("--- handle_transcribe triggered ---")
            if not audio_path:
                logger.warning("STT handler: No audio path provided (upload missing?).")
                gr.Warning("Please upload an audio file first.")
                return ""
            logger.info(f"STT handler: Processing uploaded audio='{audio_path}', lang_hint='{lang_hint}'")
            transcription = ""
            start_time = time.time()
            try:
                result = pipeline.speech_to_text(audio_path, lang_hint) # Pass the uploaded file path
                logger.info(f"Pipeline STT call took {time.time() - start_time:.3f} seconds.")
                transcription = result.get("text", "ERROR: No text found in result.")
                if "ERROR:" in transcription or result.get("error"):
                     err_msg = result.get("error", transcription); logger.error(f"STT failed: {err_msg}"); gr.Error(f"Transcription failed: {err_msg}")
                     transcription = ""
                else: logger.info("Transcription successful.")
            except Exception as e:
                logger.error(f"STT handler failed during pipeline call: {e}", exc_info=True); gr.Error(f"An unexpected error occurred: {e}")
                transcription = f"⚠️ Pipeline Error: {e}"
            logger.info(f"--- handle_transcribe finished ---")
            return transcription

        transcribe_btn.click( fn=handle_transcribe, inputs=[audio_input, src_lang], outputs=text_output )
    return tab

def create_tts_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Text-to-Speech tab (XTTS/MMS)."""
    with gr.Tab("Text → Speech") as tab:
        gr.Markdown("Synthesize speech using Coqui XTTS-v2 (en/hi) or Facebook MMS-TTS (kn).")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    text_input = gr.Textbox( label="Input Text", lines=8, placeholder="Enter text..." )
                    tts_lang = create_language_dropdown("Language", LanguageCode.ENGLISH)
                    speaker_audio = gr.Audio(
                            label="Speaker Reference (Upload WAV - XTTS Only)",
                            type="filepath", # Expect filepath from upload
                            sources=["upload"], # ONLY upload
                            visible=True
                        )
                synthesize_btn = gr.Button("Synthesize", variant="primary")
            with gr.Column(scale=1):
                with gr.Group():
                    audio_output = gr.Audio( label="Synthesized Speech", type="filepath", autoplay=False )
                    status_output = gr.Textbox(label="Status", interactive=False)

        def toggle_speaker_input(lang_code: str) -> gr.Audio:
            if lang_code == LanguageCode.KANNADA: return gr.Audio(visible=False)
            else: return gr.Audio(visible=True)
        tts_lang.change( fn=toggle_speaker_input, inputs=[tts_lang], outputs=[speaker_audio] )

        def handle_synthesize(text: str, lang: str, speaker_ref_path: Optional[str]) -> Tuple[Optional[str], str]:
            logger.info("--- handle_synthesize triggered ---")
            status_message = ""; output_audio = None
            if not text or not text.strip():
                 logger.warning("TTS handler: No text provided."); gr.Warning("Please enter text to synthesize.")
                 status_message = "⚠️ Please enter text."; return output_audio, status_message

            logger.info(f"TTS handler: Calling pipeline. lang='{lang}', speaker_ref_path='{speaker_ref_path}', text='{text[:50]}...'")
            start_time = time.time()
            try:
                result = pipeline.text_to_speech(text, lang, speaker_ref_path) # Pass filepath
                logger.info(f"Pipeline TTS call took {time.time() - start_time:.3f} seconds.")
                output_audio = result.get("audio_path"); error = result.get("error")
                if error: logger.error(f"TTS failed: {error}"); gr.Error(f"Synthesis failed: {error}"); status_message = f"⚠️ Error: {error}"
                elif output_audio: logger.info("Synthesis successful."); status_message = "✅ Synthesis successful."
                else: logger.error("TTS returned no audio path and no error."); gr.Error("Synthesis failed for an unknown reason."); status_message = "⚠️ Error: Unknown synthesis failure."
            except Exception as e:
                logger.error(f"TTS handler failed during pipeline call: {e}", exc_info=True); gr.Error(f"An unexpected error occurred: {e}")
                status_message = f"⚠️ Error: {e}"
            logger.info(f"--- handle_synthesize finished ---")
            return output_audio, status_message

        synthesize_btn.click( fn=handle_synthesize, inputs=[text_input, tts_lang, speaker_audio], outputs=[audio_output, status_output] )
    return tab

def create_translation_tab(pipeline: EchoLangPipeline) -> gr.Tab:
     # (No changes needed)
    with gr.Tab("Text Translation") as tab:
        gr.Markdown("Translate text using local IndicTrans2 models.")
        with gr.Row():
            with gr.Column(scale=1):
                source_text = gr.Textbox(label="Source Text", lines=8, placeholder="Enter text..."); src_lang = create_language_dropdown("Source Language", LanguageCode.ENGLISH)
            with gr.Column(scale=1):
                target_text = gr.Textbox(label="Translated Text", lines=8, placeholder="Translation..."); tgt_lang = create_language_dropdown("Target Language", LanguageCode.HINDI)
        translate_btn = gr.Button("Translate", variant="primary")
        def handle_translate(text: str, src: str, tgt: str) -> str:
             if not text or not text.strip(): logger.warning("Translation requested but no text provided."); gr.Warning("Please enter text to translate."); return ""
             if src == tgt: logger.warning(f"Translate src==tgt ({src}). Skipping."); gr.Info("Source and target languages are the same."); return text
             logger.info(f"Handling translation request: {src} -> {tgt}, text='{text[:50]}...'")
             try:
                 result = pipeline.translate_text(text, src, tgt); translation = result.get("translated_text", ""); error = result.get("error")
                 if error: logger.error(f"Translation failed: {error}"); gr.Error(f"Translation failed: {error}"); return f"⚠️ Error: {error}"
                 elif translation is not None: logger.info("Translation successful."); return translation
                 else: logger.warning("Translation successful but result is None/Empty string."); return ""
             except Exception as e: logger.error(f"Translation handler failed: {e}", exc_info=True); gr.Error(f"An unexpected error occurred: {e}"); return f"⚠️ Error: {e}"
        translate_btn.click( fn=handle_translate, inputs=[source_text, src_lang, tgt_lang], outputs=target_text )
    return tab

def create_speech_to_translated_text_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Speech to Translated Text tab."""
    with gr.Tab("Speech → Translated Text") as tab:
        gr.Markdown("Transcribe **uploaded** audio (FasterWhisper), then translate (IndicTrans2).")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech (Upload)", type="filepath", sources=["upload"] # ONLY upload
                )
                src_lang = create_language_dropdown("Source Language", LanguageCode.ENGLISH)
                tgt_lang = create_language_dropdown("Target Language", LanguageCode.HINDI)
                process_btn = gr.Button("Process Speech to Translated Text", variant="primary")
            with gr.Column(scale=2):
                 original_text = gr.Textbox( label="Original Transcription", lines=5, placeholder="Transcription..." )
                 translated_text = gr.Textbox( label="Translated Text", lines=5, placeholder="Translation..." )
                 status_output = gr.Textbox(label="Status", interactive=False)

        def handle_speech_to_translated_text(audio_path: Optional[str], src: str, tgt: str) -> Tuple[str, str, str]:
            logger.info("--- handle_speech_to_translated_text triggered ---")
            status = ""; original = ""; translated = ""
            if not audio_path: # Check upload
                gr.Warning("Please upload audio first.")
                return original, translated, "⚠️ Please provide audio."
            logger.info(f"S2TT handler: Processing uploaded audio='{audio_path}', {src} -> {tgt}")
            start_time = time.time()
            try:
                result = pipeline.speech_to_translated_text(audio_path, src, tgt)
                logger.info(f"Pipeline S2TT call took {time.time() - start_time:.3f} seconds.")
                transcription_res = result.get("transcription", {}); translation_res = result.get("translation", {})
                original = transcription_res.get("text", ""); translated = translation_res.get("translated_text", "")
                transcription_error_msg = transcription_res.get("error"); translation_error_msg = translation_res.get("error")
                transcription_text_error = original.startswith("ERROR:") if isinstance(original, str) else False; translation_text_error = translated.startswith("ERROR:") if isinstance(translated, str) else False
                final_error = transcription_error_msg or (original if transcription_text_error else None) or translation_error_msg or (translated if translation_text_error else None)
                if final_error: status = f"⚠️ Error during processing: {final_error}"; logger.error(status); gr.Error(status);
                elif src == tgt and original: status = "✅ Transcription complete (translation not needed)."; translated = original
                elif original and translated is not None: status = "✅ Processing complete."; logger.info(status)
                else: status = "⚠️ Processing failed for an unknown reason (check logs)."; logger.error(status + f" Results: {result}"); gr.Error(status)
            except Exception as e:
                logger.error(f"Speech -> Translated Text handler failed: {e}", exc_info=True); status = f"⚠️ Unexpected Error: {e}"; gr.Error(status)
                original = ""; translated = ""
            logger.info(f"--- handle_speech_to_translated_text finished ---")
            return original, translated, status

        process_btn.click( fn=handle_speech_to_translated_text, inputs=[audio_input, src_lang, tgt_lang], outputs=[original_text, translated_text, status_output] )
    return tab

def create_speech_to_translated_speech_tab(pipeline: EchoLangPipeline) -> gr.Tab:
    """Create Speech to Translated Speech tab."""
    with gr.Tab("Speech → Translated Speech") as tab:
        gr.Markdown("Transcribe **uploaded** audio (FasterWhisper), translate (IndicTrans2), synthesize (XTTS/MMS).")
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Input Speech (Upload)", type="filepath", sources=["upload"] # ONLY upload
                )
                src_lang = create_language_dropdown("Source Language", LanguageCode.ENGLISH)
                tgt_lang = create_language_dropdown("Target Language", LanguageCode.HINDI)
                speaker_audio = gr.Audio(
                    label="Output Speaker Ref (Upload WAV - XTTS Only)", type="filepath", sources=["upload"], # ONLY upload
                    visible=(tgt_lang != LanguageCode.KANNADA)
                )
                process_btn = gr.Button("Process Speech to Translated Speech", variant="primary")
            with gr.Column(scale=2):
                 original_text = gr.Textbox( label="Original Transcription", lines=4, placeholder="..." )
                 translated_text = gr.Textbox( label="Translated Text", lines=4, placeholder="..." )
                 audio_output = gr.Audio( label="Translated Speech", type="filepath", autoplay=False )
                 status_output = gr.Textbox(label="Status", interactive=False)

        def toggle_speaker_input_s2s(target_lang_code: str) -> gr.Audio:
            if target_lang_code == LanguageCode.KANNADA: return gr.Audio(visible=False)
            else: return gr.Audio(visible=True)
        tgt_lang.change( fn=toggle_speaker_input_s2s, inputs=[tgt_lang], outputs=[speaker_audio] )

        def handle_speech_to_translated_speech(audio_path: Optional[str], src: str, tgt: str, speaker_ref_path: Optional[str]) -> Tuple[str, str, Optional[str], str]:
            logger.info("--- handle_speech_to_translated_speech triggered ---")
            status = ""; original = ""; translated = ""; output_audio = None
            if not audio_path: # Check main audio upload
                gr.Warning("Please upload input audio first.")
                return original, translated, output_audio, "⚠️ Please provide input audio."
            if tgt == LanguageCode.KANNADA: speaker_ref_path = None
            logger.info(f"S2S handler: Calling pipeline. audio_path='{audio_path}', {src} -> {tgt}, speaker_ref_path='{speaker_ref_path}'")
            start_time = time.time()
            try:
                result = pipeline.speech_to_translated_speech(audio_path, src, tgt, speaker_ref_path) # Pass filepaths
                logger.info(f"Pipeline S2S call took {time.time() - start_time:.3f} seconds.")
                transcription_res = result.get("transcription", {}); translation_res = result.get("translation", {}); synthesis_res = result.get("synthesis", {})
                original = transcription_res.get("text", ""); translated = translation_res.get("translated_text", ""); output_audio = synthesis_res.get("audio_path")
                transcription_error_msg = transcription_res.get("error"); translation_error_msg = translation_res.get("error"); synthesis_error_msg = synthesis_res.get("error")
                transcription_text_error = original.startswith("ERROR:") if isinstance(original, str) else False; translation_text_error = translated.startswith("ERROR:") if isinstance(translated, str) else False
                final_error = transcription_error_msg or (original if transcription_text_error else None) or translation_error_msg or (translated if translation_text_error else None) or synthesis_error_msg
                if final_error: status = f"⚠️ Error during processing: {final_error}"; logger.error(status); gr.Error(status);
                elif src == tgt and original and output_audio: status = "✅ Transcription and resynthesis complete."; translated = original
                elif original and translated is not None and output_audio: status = "✅ Processing complete."; logger.info(status)
                else: status = "⚠️ Processing failed for an unknown reason (check logs)."; logger.error(status + f" Results: {result}"); gr.Error(status)
            except Exception as e:
                logger.error(f"Speech -> Translated Speech handler failed: {e}", exc_info=True); status = f"⚠️ Unexpected Error: {e}"; gr.Error(status)
                original = ""; translated = ""; output_audio = None
            logger.info(f"--- handle_speech_to_translated_speech finished ---")
            return original, translated, output_audio, status

        process_btn.click(
            fn=handle_speech_to_translated_speech,
            inputs=[audio_input, src_lang, tgt_lang, speaker_audio],
            outputs=[original_text, translated_text, audio_output, status_output]
        )
    return tab

# --- NEW: Function to create the Recording Info Tab ---
def create_recording_tab() -> gr.Tab:
    """Create the Recording Info tab."""
    with gr.Tab("🎙️ Record Audio (External Script)") as tab:
        gr.Markdown(
            """
            ## Recording Audio Separately

            Due to browser inconsistencies, microphone recording directly in this interface
            can be unreliable.

            **Recommended Workflow:**

            1.  **Run the Recording Script:**
                * Open your terminal (ensure your `venv` is activated: `source venv/bin/activate`).
                * Run the command: `python record_audio.py`
                * Follow the prompts in the terminal to select a microphone (optional), start/stop recording, and name your file.
                * Your recording will be saved as a `.wav` file in the `recordings/` folder within your project directory.

            2.  **Upload the File:**
                * Go to the desired tab in *this* web UI (e.g., "Speech → Text", "Text → Speech" for speaker reference, etc.).
                * Use the **"Upload File"** option in the relevant audio input box to select the `.wav` file you just saved in the `recordings/` folder.

            3.  **Process:**
                * Select other options (like languages) and click the main button (e.g., "Transcribe", "Synthesize", "Process...").

            This ensures the application receives a clean, compatible audio file for processing.
            """
        )
        # Optionally add a File Explorer output to show the recordings folder?
        # file_output = gr.FileExplorer(root_dir="recordings", file_count="multiple", glob="**/*.wav", label="Recent Recordings (in ./recordings/)")
        # tab.load(lambda: None, None, file_output) # Refresh on load? Might not work well.

    return tab