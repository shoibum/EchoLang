# src/web/app.py
import gradio as gr
import os
from pathlib import Path
import logging
import sys

# Use relative imports
from ..pipeline import EchoLangPipeline
# --- IMPORT New Tab Function ---
from .components import (
    create_stt_tab,
    create_tts_tab,
    create_translation_tab,
    create_speech_to_translated_text_tab,
    create_speech_to_translated_speech_tab,
    create_recording_tab # Import the new function
)
# --- END IMPORT ---
try:
    from .. import config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

# (Keep get_pipeline function as is)
pipeline_instance = None
def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        logger.info("Initializing global EchoLangPipeline instance...")
        try: pipeline_instance = EchoLangPipeline(); logger.info("Global EchoLangPipeline instance initialized successfully.")
        except Exception as e: logger.error(f"Failed to initialize global pipeline: {e}", exc_info=True); raise RuntimeError(f"Failed to initialize the backend pipeline: {e}") from e
    return pipeline_instance

def create_app():
    """Create the Gradio web application using settings from config."""
    logger.info("Creating Gradio App interface...")

    try:
        pipeline = get_pipeline()
    except RuntimeError as e:
         # (Keep error handling UI as is)
         logger.error("Pipeline initialization failed. Creating error UI.")
         with gr.Blocks(title="EchoLang - Error", theme=config.GRADIO_THEME) as app:
               gr.Markdown(f"""<h1>EchoLang Initialization Error</h1><p>Could not initialize the application backend. Please check the logs.</p><p><strong>Error:</strong> <code>{e}</code></p>""")
         return app

    with gr.Blocks(title="EchoLang", theme=config.GRADIO_THEME) as app:
        # Minimal Header
        gr.HTML("<h1 style='text-align: center; margin-bottom: 1rem'>EchoLang</h1>")

        # --- Tabs (Add Recording Tab First) ---
        with gr.Tabs():
            logger.debug("Creating Recording tab...")
            create_recording_tab() # Add the new tab here

            logger.debug("Creating STT tab...")
            create_stt_tab(pipeline)

            logger.debug("Creating TTS tab...")
            create_tts_tab(pipeline)

            logger.debug("Creating Translation tab...")
            create_translation_tab(pipeline)

            logger.debug("Creating Speech -> Translated Text tab...")
            create_speech_to_translated_text_tab(pipeline)

            logger.debug("Creating Speech -> Translated Speech tab...")
            create_speech_to_translated_speech_tab(pipeline)
        # --- End Tabs ---

        # --- Keep "About" Section ---
        about_html = f"""
        <hr> <h3>About This Application</h3> <p>{config.GRADIO_DESCRIPTION.replace('<0xF0><0x9F><0x94><0x8A>', '🔊')}</p>
        <ul><li><strong>Device:</strong> This app runs locally on your <strong>{config.APP_DEVICE.upper()}</strong>.</li>
        <li><strong>Models Used:</strong><ul>
        <li><strong>STT:</strong> FasterWhisper (Fine-tuned kn/hi, Base en - CPU)</li>
        <li><strong>Translation:</strong> IndicTrans2 Distilled (Local - CPU)</li>
        <li><strong>TTS:</strong> Coqui XTTS-v2 (en/hi - CPU) / Facebook MMS-TTS (kn - CPU)</li></ul></li>
        <li><strong>Model Storage:</strong> Models are downloaded to Hugging Face cache or <code>./models</code>. Check console logs for details.</li></ul>
        """
        gr.HTML(value=about_html)
        # --- End of About section ---

    logger.info("Gradio App interface created successfully.")
    return app

# (Keep launch_app function as is)
def launch_app(share: bool = False, server_port: int = 7860):
    try: app = create_app(); logger.info(f"Launching Gradio app on port {server_port}. Share={share}"); app.launch(share=share, server_port=server_port)
    except Exception as e: logger.error(f"Failed to create or launch the Gradio app: {e}", exc_info=True); print(f"\nFATAL: Failed to launch EchoLang UI. Check logs for details.\nError: {e}\n", file=sys.stderr); sys.exit(1)