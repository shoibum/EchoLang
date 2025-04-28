# src/web/app.py
import gradio as gr
import os
from pathlib import Path
import logging
import sys

# Use relative imports for pipeline, model_utils, components, and config
from ..pipeline import EchoLangPipeline
# from ..utils.model_utils import ModelManager # Not needed directly here now
from .components import (
    create_stt_tab,
    create_tts_tab,
    create_translation_tab,
    create_speech_to_translated_text_tab,
    create_speech_to_translated_speech_tab
)
try:
    from .. import config
except ImportError:
    # Allow running directly for testing? Unlikely needed if main.py is entry point
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

# Configure logging (basic example, consider moving to main.py or a dedicated logging setup)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Setup logger (assuming config elsewhere)

# Global pipeline instance (lazy initialization)
pipeline_instance = None

def get_pipeline():
    """Lazy initializes and returns the global pipeline instance."""
    global pipeline_instance
    if pipeline_instance is None:
        logger.info("Initializing global EchoLangPipeline instance...")
        try:
            # ModelManager is created internally by Pipeline now if not passed
            pipeline_instance = EchoLangPipeline()
            logger.info("Global EchoLangPipeline instance initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize global pipeline: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize the backend pipeline: {e}") from e
    return pipeline_instance

def create_app():
    """Create the Gradio web application using settings from config."""
    logger.info("Creating Gradio App interface...")

    try:
        pipeline = get_pipeline()
    except RuntimeError as e:
         logger.error("Pipeline initialization failed. Creating error UI.")
         with gr.Blocks(title="EchoLang - Error", theme=gr.themes.Soft()) as app:
               gr.Markdown(f"""
               # EchoLang Initialization Error
               Could not initialize the application backend. Please check the logs.
               **Error:** `{e}`
               """)
         return app

    with gr.Blocks(title=config.GRADIO_TITLE, theme=config.GRADIO_THEME) as app:
        gr.Markdown(
            f"""
            # {config.GRADIO_TITLE}

            {config.GRADIO_DESCRIPTION}

            This app runs locally on your device ({config.APP_DEVICE}). Models are downloaded to the Hugging Face cache or `./models`.
            """
        )

        with gr.Tabs():
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

        # --- Updated "About" section ---
        gr.Markdown(
            f"""
            ---
            ### About EchoLang Models

            * **Speech-to-Text (STT):** FasterWhisper (Fine-tuned kn/hi, Base en models via CTranslate2)
            * **Machine Translation (MT):** Meta NLLB ({config.DEFAULT_NLLB_MODEL_KEY}) via Transformers
            * **Text-to-Speech (TTS):** Coqui XTTS-v2 via TTS library

            `transformers` models (NLLB) are cached in `~/.cache/huggingface/hub`.
            FasterWhisper models (CTranslate2 format) used here are stored locally in `{config.MODELS_DIR}`.
            XTTS-v2 model files are stored in `{config.MODELS_DIR / 'xtts_v2'}`.
            Check console logs for download/conversion progress and status.
            """
        )
        # --- End of updated section ---

    logger.info("Gradio App interface created successfully.")
    return app

def launch_app(share: bool = False, server_port: int = 7860):
    """Launch the Gradio app."""
    try:
        app = create_app()
        logger.info(f"Launching Gradio app on port {server_port}. Share={share}")
        # Can add server_name="0.0.0.0" to allow access from other devices on network
        app.launch(share=share, server_port=server_port)
    except Exception as e:
         logger.error(f"Failed to create or launch the Gradio app: {e}", exc_info=True)
         print(f"\nFATAL: Failed to launch EchoLang UI. Check logs for details.\nError: {e}\n", file=sys.stderr)
         sys.exit(1)

# No main guard here, main.py is entry point