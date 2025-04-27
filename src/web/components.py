# src/web/components.py
import gradio as gr
from typing import Dict, List, Optional, Callable
from pathlib import Path

from ..utils.language import LanguageCode

def create_stt_tab(pipeline_fn: Callable) -> gr.Tab:
    """Create Speech-to-Text tab."""
    with gr.Tab("Speech → Text") as tab:
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Input Speech",
                    type="filepath",
                    sources=["microphone", "upload"]
                )
                src_lang = gr.Dropdown(
                    label="Source Language",
                    choices=[
                        (LanguageCode.code_to_name(LanguageCode.ENGLISH), LanguageCode.ENGLISH),
                        (LanguageCode.code_to_name(LanguageCode.HINDI), LanguageCode.HINDI),
                        (LanguageCode.code_to_name(LanguageCode.KANNADA), LanguageCode.KANNADA)
                    ],
                    value=LanguageCode.ENGLISH
                )
                transcribe_btn = gr.Button("Transcribe", variant="primary")
                
            with gr.Column():
                text_output = gr.Textbox(
                    label="Transcription",
                    lines=10,
                    placeholder="Transcribed text will appear here..."
                )
                
        transcribe_btn.click(
            fn=lambda audio, lang: pipeline_fn.speech_to_text(audio, lang)["text"] 
               if audio else "Please record or upload audio first",
            inputs=[audio_input, src_lang],
            outputs=text_output
        )
        
    return tab

def create_tts_tab(pipeline_fn: Callable) -> gr.Tab:
    """Create Text-to-Speech tab."""
    with gr.Tab("Text → Speech") as tab:
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=8,
                    placeholder="Enter text to synthesize..."
                )
                tts_lang = gr.Dropdown(
                    label="Language",
                    choices=[
                        (LanguageCode.code_to_name(LanguageCode.ENGLISH), LanguageCode.ENGLISH),
                        (LanguageCode.code_to_name(LanguageCode.HINDI), LanguageCode.HINDI),
                        (LanguageCode.code_to_name(LanguageCode.KANNADA), LanguageCode.KANNADA)
                    ],
                    value=LanguageCode.ENGLISH
                )
                speed_slider = gr.Slider(
                    label="Speed",
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.1
                )
                speaker_audio = gr.Audio(
                    label="Speaker Reference (Optional)",
                    type="filepath",
                    sources=["microphone", "upload"],
                    visible=True
                )
                synthesize_btn = gr.Button("Synthesize", variant="primary")
                
            with gr.Column():
                audio_output = gr.Audio(
                    label="Synthesized Speech",
                    type="filepath"
                )
                
        synthesize_btn.click(
            fn=lambda text, lang, speed, spk: pipeline_fn.text_to_speech(
                text, lang, spk, speed
            )["audio_path"] if text.strip() else None,
            inputs=[text_input, tts_lang, speed_slider, speaker_audio],
            outputs=audio_output
        )
        
    return tab

def create_translation_tab(pipeline_fn: Callable) -> gr.Tab:
    """Create Text Translation tab."""
    with gr.Tab("Text Translation") as tab:
        with gr.Row():
            with gr.Column():
                source_text = gr.Textbox(
                    label="Source Text",
                    lines=8,
                    placeholder="Enter text to translate..."
                )
                src_lang = gr.Dropdown(
                    label="Source Language",
                    choices=[
                        (LanguageCode.code_to_name(LanguageCode.ENGLISH), LanguageCode.ENGLISH),
                        (LanguageCode.code_to_name(LanguageCode.HINDI), LanguageCode.HINDI),
                        (LanguageCode.code_to_name(LanguageCode.KANNADA), LanguageCode.KANNADA)
                    ],
                    value=LanguageCode.ENGLISH
                )
                
            with gr.Column():
                target_text = gr.Textbox(
                    label="Translated Text",
                    lines=8,
                    placeholder="Translation will appear here..."
                )
                tgt_lang = gr.Dropdown(
                    label="Target Language",
                    choices=[
                        (LanguageCode.code_to_name(LanguageCode.ENGLISH), LanguageCode.ENGLISH),
                        (LanguageCode.code_to_name(LanguageCode.HINDI), LanguageCode.HINDI),
                        (LanguageCode.code_to_name(LanguageCode.KANNADA), LanguageCode.KANNADA)
                    ],
                    value=LanguageCode.HINDI
                )
                
        translate_btn = gr.Button("Translate", variant="primary")
        
        translate_btn.click(
            fn=lambda text, src, tgt: pipeline_fn.translate_text(
                text, src, tgt
            )["translated_text"] if text.strip() else "Please enter text to translate",
            inputs=[source_text, src_lang, tgt_lang],
            outputs=target_text
        )
        
    return tab

def create_speech_to_translated_text_tab(pipeline_fn: Callable) -> gr.Tab:
    """Create Speech to Translated Text tab."""
    with gr.Tab("Speech → Translated Text") as tab:
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Input Speech",
                    type="filepath",
                    sources=["microphone", "upload"]
                )
                src_lang = gr.Dropdown(
                    label="Source Language",
                    choices=[
                        (LanguageCode.code_to_name(LanguageCode.ENGLISH), LanguageCode.ENGLISH),
                        (LanguageCode.code_to_name(LanguageCode.HINDI), LanguageCode.HINDI),
                        (LanguageCode.code_to_name(LanguageCode.KANNADA), LanguageCode.KANNADA)
                    ],
                    value=LanguageCode.ENGLISH
                )
                
            with gr.Column():
                translated_text = gr.Textbox(
                    label="Translated Text",
                    lines=8,
                    placeholder="Translated text will appear here..."
                )
                tgt_lang = gr.Dropdown(
                    label="Target Language",
                    choices=[
                        (LanguageCode.code_to_name(LanguageCode.ENGLISH), LanguageCode.ENGLISH),
                        (LanguageCode.code_to_name(LanguageCode.HINDI), LanguageCode.HINDI),
                        (LanguageCode.code_to_name(LanguageCode.KANNADA), LanguageCode.KANNADA)
                    ],
                    value=LanguageCode.HINDI
                )
                
        original_text = gr.Textbox(
            label="Original Transcription",
            lines=4,
            placeholder="Original transcription will appear here..."
        )
        
        process_btn = gr.Button("Process", variant="primary")
        
        process_btn.click(
            fn=lambda audio, src, tgt: {
                "transcription": pipeline_fn.speech_to_translated_text(
                    audio, src, tgt
                )["transcription"]["text"] if audio else "Please record or upload audio first",
                "translation": pipeline_fn.speech_to_translated_text(
                    audio, src, tgt
                )["translation"]["translated_text"] if audio else ""
            },
            inputs=[audio_input, src_lang, tgt_lang],
            outputs=[original_text, translated_text]
        )
        
    return tab

def create_speech_to_translated_speech_tab(pipeline_fn: Callable) -> gr.Tab:
    """Create Speech to Translated Speech tab."""
    with gr.Tab("Speech → Translated Speech") as tab:
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Input Speech",
                    type="filepath",
                    sources=["microphone", "upload"]
                )
                src_lang = gr.Dropdown(
                    label="Source Language",
                    choices=[
                        (LanguageCode.code_to_name(LanguageCode.ENGLISH), LanguageCode.ENGLISH),
                        (LanguageCode.code_to_name(LanguageCode.HINDI), LanguageCode.HINDI),
                        (LanguageCode.code_to_name(LanguageCode.KANNADA), LanguageCode.KANNADA)
                    ],
                    value=LanguageCode.ENGLISH
                )
                tgt_lang = gr.Dropdown(
                    label="Target Language",
                    choices=[
                        (LanguageCode.code_to_name(LanguageCode.ENGLISH), LanguageCode.ENGLISH),
                        (LanguageCode.code_to_name(LanguageCode.HINDI), LanguageCode.HINDI),
                        (LanguageCode.code_to_name(LanguageCode.KANNADA), LanguageCode.KANNADA)
                    ],
                    value=LanguageCode.HINDI
                )
                speed_slider = gr.Slider(
                    label="Output Speed",
                    minimum=0.5,
                    maximum=1.5,
                    value=1.0,
                    step=0.1
                )
                
            with gr.Column():
                audio_output = gr.Audio(
                    label="Translated Speech",
                    type="filepath"
                )
                
        with gr.Row():
            original_text = gr.Textbox(
                label="Original Transcription",
                lines=4,
                placeholder="Original transcription will appear here..."
            )
            translated_text = gr.Textbox(
                label="Translated Text",
                lines=4,
                placeholder="Translated text will appear here..."
            )
        
        process_btn = gr.Button("Process", variant="primary")
        
        def process_speech_translation(audio, src, tgt, speed):
            if not audio:
                return "Please record or upload audio first", "", None
                
            result = pipeline_fn.speech_to_translated_speech(audio, src, tgt, None, speed)
            
            return (
                result["transcription"]["text"],
                result["translation"]["translated_text"],
                result["synthesis"]["audio_path"]
            )
        
        process_btn.click(
            fn=process_speech_translation,
            inputs=[audio_input, src_lang, tgt_lang, speed_slider],
            outputs=[original_text, translated_text, audio_output]
        )
        
    return tab