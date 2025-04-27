# src/web/app.py
import gradio as gr
import os
from pathlib import Path

from ..pipeline import EchoLangPipeline
from ..utils.model_utils import ModelManager
from .components import (
    create_stt_tab,
    create_tts_tab,
    create_translation_tab,
    create_speech_to_translated_text_tab,
    create_speech_to_translated_speech_tab
)

def create_app(pipeline: EchoLangPipeline = None):
    """Create Gradio web application."""
    # Initialize pipeline if not provided
    if pipeline is None:
        pipeline = EchoLangPipeline()
    
    # Create the Gradio interface
    with gr.Blocks(title="EchoLang", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # EchoLang – GPU‑Accelerated Multilingual Speech ↔ Text ↔ Speech
            
            **English · हिंदी · ಕನ್ನಡ**
            
            EchoLang runs entirely on the Apple‑Silicon GPU (Metal/MPS) and lets you:
            * transcribe speech,
            * translate it between English, Hindi, and Kannada,
            * speak the result back — all in one click.
            """
        )
        
        with gr.Tabs():
            # Create all the tabs
            stt_tab = create_stt_tab(pipeline)
            tts_tab = create_tts_tab(pipeline)
            translation_tab = create_translation_tab(pipeline)
            stt_translation_tab = create_speech_to_translated_text_tab(pipeline)
            speech_translation_tab = create_speech_to_translated_speech_tab(pipeline)
            
        gr.Markdown(
            """
            ### About EchoLang
            
            This application uses:
            * **Speech‑to‑Text (STT)** — Seamless M4T model
            * **Machine Translation** — IndicTrans‑2 model
            * **Text‑to‑Speech (TTS)** — XTTS‑v2 model
            
            All models run locally on your device using GPU acceleration.
            """
        )
            
    return app

def launch_app(share=False, server_port=7860):
    """Launch the Gradio app."""
    app = create_app()
    app.launch(share=share, server_port=server_port)
    
if __name__ == "__main__":
    launch_app()