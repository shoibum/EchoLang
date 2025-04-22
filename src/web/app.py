"""
Gradio web interface for the multilingual STT/TTS application.
"""

import os
import tempfile
import numpy as np
import gradio as gr

from src.pipeline import Pipeline
from src.config import LANGUAGES, GRADIO_TITLE, GRADIO_DESCRIPTION, GRADIO_THEME

class WebApp:
    """
    Web application class using Gradio.
    """
    
    def __init__(self):
        """Initialize the web application."""
        self.pipeline = Pipeline()
    
    def speech_to_text_fn(
        self, 
        audio, 
        source_language
    ):
        """Speech to text function for Gradio."""
        if audio is None:
            return "Please record or upload audio."
        
        # Handle gradio audio format
        if isinstance(audio, tuple):
            # Convert sample rate and audio array to bytes
            sample_rate, audio_array = audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                import soundfile as sf
                sf.write(temp_filename, audio_array, sample_rate)
            
            with open(temp_filename, "rb") as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_filename)
        else:
            # Assume it's a filepath
            with open(audio, "rb") as f:
                audio_bytes = f.read()
        
        # Use None for language detection if "Auto-detect" is selected
        lang_code = None if source_language == "Auto-detect" else source_language
        
        # Process with pipeline
        result = self.pipeline.speech_to_text(audio_bytes, lang_code)
        
        # Format output
        detected_lang = result.get("language", "unknown")
        language_name = LANGUAGES.get(detected_lang, "Unknown")
        
        output = f"Detected Language: {language_name} ({detected_lang})\n\n{result['text']}"
        return output
    
    def text_to_speech_fn(
        self, 
        text, 
        language
    ):
        """Text to speech function for Gradio."""
        if not text:
            return None
        
        # Process with pipeline
        audio_bytes = self.pipeline.text_to_speech(text, language)
        
        # Create temporary WAV file that Gradio can use
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            if isinstance(audio_bytes, bytes):
                temp_file.write(audio_bytes)
            else:
                # If it's a path, copy the content
                with open(audio_bytes, "rb") as src_file:
                    temp_file.write(src_file.read())
        
        return temp_filename
    
    def translate_text_fn(
        self, 
        text, 
        source_language, 
        target_language
    ):
        """Text translation function for Gradio."""
        if not text:
            return ""
        
        # Process with pipeline
        translated = self.pipeline.translate_text(text, source_language, target_language)
        return translated
    
    def speech_to_translated_text_fn(
        self, 
        audio, 
        source_language, 
        target_language
    ):
        """Speech to translated text function for Gradio."""
        if audio is None:
            return "Please record or upload audio."
        
        # Handle gradio audio format
        if isinstance(audio, tuple):
            # Convert sample rate and audio array to bytes
            sample_rate, audio_array = audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                import soundfile as sf
                sf.write(temp_filename, audio_array, sample_rate)
            
            with open(temp_filename, "rb") as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_filename)
        else:
            # Assume it's a filepath
            with open(audio, "rb") as f:
                audio_bytes = f.read()
        
        # Use None for language detection if "Auto-detect" is selected
        src_lang_code = None if source_language == "Auto-detect" else source_language
        
        # Process with pipeline
        result = self.pipeline.speech_to_translated_text(
            audio_bytes, src_lang_code, target_language
        )
        
        # Format output
        detected_lang = result.get("source_language", "unknown")
        source_language_name = LANGUAGES.get(detected_lang, "Unknown")
        target_language_name = LANGUAGES.get(target_language, "Unknown")
        
        output = f"Source Language: {source_language_name} ({detected_lang})\n"
        output += f"Target Language: {target_language_name} ({target_language})\n\n"
        output += f"Original: {result['transcription']}\n\n"
        output += f"Translation: {result['translation']}"
        
        return output
    
    def speech_to_translated_speech_fn(
        self, 
        audio, 
        source_language, 
        target_language
    ):
        """Speech to translated speech function for Gradio."""
        if audio is None:
            return "Please record or upload audio.", None
        
        # Handle gradio audio format
        if isinstance(audio, tuple):
            # Convert sample rate and audio array to bytes
            sample_rate, audio_array = audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                import soundfile as sf
                sf.write(temp_filename, audio_array, sample_rate)
            
            with open(temp_filename, "rb") as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_filename)
        else:
            # Assume it's a filepath
            with open(audio, "rb") as f:
                audio_bytes = f.read()
        
        # Use None for language detection if "Auto-detect" is selected
        src_lang_code = None if source_language == "Auto-detect" else source_language
        
        # Process with pipeline
        result, audio_output = self.pipeline.speech_to_translated_speech(
            audio_bytes, src_lang_code, target_language
        )
        
        # Format text output
        detected_lang = result.get("source_language", "unknown")
        source_language_name = LANGUAGES.get(detected_lang, "Unknown")
        target_language_name = LANGUAGES.get(target_language, "Unknown")
        
        output = f"Source Language: {source_language_name} ({detected_lang})\n"
        output += f"Target Language: {target_language_name} ({target_language})\n\n"
        output += f"Original: {result['transcription']}\n\n"
        output += f"Translation: {result['translation']}"
        
        # Create temporary WAV file that Gradio can use for audio output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_filename = temp_file.name
            if isinstance(audio_output, bytes):
                temp_file.write(audio_output)
            else:
                # If it's a path, copy the content
                with open(audio_output, "rb") as src_file:
                    temp_file.write(src_file.read())
        
        return output, temp_filename
    
    def create_app(self):
        """Create and configure the Gradio application."""
        # Prepare language choices
        language_choices = ["Auto-detect"] + list(LANGUAGES.keys())
        language_names = {code: LANGUAGES[code] for code in LANGUAGES}
        language_names["Auto-detect"] = "Auto-detect"
        
        with gr.Blocks(title=GRADIO_TITLE, theme=GRADIO_THEME) as app:
            gr.Markdown(f"# {GRADIO_TITLE}")
            gr.Markdown(GRADIO_DESCRIPTION)
            
            with gr.Tabs():
                # Speech to Text Tab
                with gr.TabItem("Speech to Text"):
                    with gr.Row():
                        with gr.Column():
                            stt_audio_input = gr.Audio(
                                label="Audio Input",
                                type="filepath"
                            )
                            stt_source_lang = gr.Dropdown(
                                choices=language_choices,
                                value="Auto-detect",
                                label="Source Language"
                            )
                            stt_button = gr.Button("Transcribe")
                        
                        with gr.Column():
                            stt_output = gr.Textbox(
                                label="Transcription",
                                lines=5
                            )
                    
                    # Set up event handler
                    stt_button.click(
                        fn=self.speech_to_text_fn,
                        inputs=[stt_audio_input, stt_source_lang],
                        outputs=stt_output
                    )
                
                # Text to Speech Tab
                with gr.TabItem("Text to Speech"):
                    with gr.Row():
                        with gr.Column():
                            tts_text_input = gr.Textbox(
                                label="Text Input",
                                lines=5
                            )
                            tts_lang = gr.Dropdown(
                                choices=list(LANGUAGES.keys()),
                                value="en",
                                label="Language"
                            )
                            tts_button = gr.Button("Synthesize")
                        
                        with gr.Column():
                            tts_output = gr.Audio(
                                label="Synthesized Speech"
                            )
                    
                    # Set up event handler
                    tts_button.click(
                        fn=self.text_to_speech_fn,
                        inputs=[tts_text_input, tts_lang],
                        outputs=tts_output
                    )
                
                # Text Translation Tab
                with gr.TabItem("Text Translation"):
                    with gr.Row():
                        with gr.Column():
                            trans_text_input = gr.Textbox(
                                label="Text Input",
                                lines=5
                            )
                            with gr.Row():
                                trans_source_lang = gr.Dropdown(
                                    choices=list(LANGUAGES.keys()),
                                    value="en",
                                    label="Source Language"
                                )
                                trans_target_lang = gr.Dropdown(
                                    choices=list(LANGUAGES.keys()),
                                    value="hi",
                                    label="Target Language"
                                )
                            trans_button = gr.Button("Translate")
                        
                        with gr.Column():
                            trans_output = gr.Textbox(
                                label="Translation",
                                lines=5
                            )
                    
                    # Set up event handler
                    trans_button.click(
                        fn=self.translate_text_fn,
                        inputs=[trans_text_input, trans_source_lang, trans_target_lang],
                        outputs=trans_output
                    )
                
                # Speech to Translated Text Tab
                with gr.TabItem("Speech to Translated Text"):
                    with gr.Row():
                        with gr.Column():
                            st_audio_input = gr.Audio(
                                label="Audio Input", 
                                type="filepath"
                            )
                            with gr.Row():
                                st_source_lang = gr.Dropdown(
                                    choices=language_choices,
                                    value="Auto-detect",
                                    label="Source Language"
                                )
                                st_target_lang = gr.Dropdown(
                                    choices=list(LANGUAGES.keys()),
                                    value="hi",
                                    label="Target Language"
                                )
                            st_button = gr.Button("Transcribe and Translate")
                        
                        with gr.Column():
                            st_output = gr.Textbox(
                                label="Translation Result",
                                lines=10
                            )
                    
                    # Set up event handler
                    st_button.click(
                        fn=self.speech_to_translated_text_fn,
                        inputs=[st_audio_input, st_source_lang, st_target_lang],
                        outputs=st_output
                    )
                
                # Speech to Translated Speech Tab
                with gr.TabItem("Speech to Translated Speech"):
                    with gr.Row():
                        with gr.Column():
                            sts_audio_input = gr.Audio(
                                label="Audio Input", 
                                type="filepath"
                            )
                            with gr.Row():
                                sts_source_lang = gr.Dropdown(
                                    choices=language_choices,
                                    value="Auto-detect",
                                    label="Source Language"
                                )
                                sts_target_lang = gr.Dropdown(
                                    choices=list(LANGUAGES.keys()),
                                    value="hi",
                                    label="Target Language"
                                )
                            sts_button = gr.Button("Translate Speech")
                        
                        with gr.Column():
                            sts_text_output = gr.Textbox(
                                label="Translation Details",
                                lines=10
                            )
                            sts_audio_output = gr.Audio(
                                label="Translated Speech"
                            )
                    
                    # Set up event handler
                    sts_button.click(
                        fn=self.speech_to_translated_speech_fn,
                        inputs=[sts_audio_input, sts_source_lang, sts_target_lang],
                        outputs=[sts_text_output, sts_audio_output]
                    )
                
                # About Tab
                with gr.TabItem("About"):
                    gr.Markdown("""
                    # About This Project
                    
                    This is a multilingual Speech-to-Text and Text-to-Speech application that supports English, Hindi, and Kannada.
                    
                    ## Features:
                    
                    - **Speech to Text**: Transcribe speech in English, Hindi, and Kannada.
                    - **Text to Speech**: Convert text to speech in all three languages.
                    - **Text Translation**: Translate text between the supported languages.
                    - **Speech to Translated Text**: Transcribe speech and translate it to another language.
                    - **Speech to Translated Speech**: Translate speech from one language to another.
                    
                    ## Technologies Used:
                    
                    - **OpenAI Whisper**: For speech recognition.
                    - **Hugging Face Transformers**: For text translation.
                    - **Coqui TTS**: For text-to-speech synthesis.
                    - **Gradio**: For the web interface.
                    
                    ## Project By:
                    
                    Final Year Computer Science Engineering Project
                    """)
        
        return app
    
    def launch(self, **kwargs):
        """Launch the Gradio application."""
        app = self.create_app()
        app.launch(**kwargs)

# Function to launch the app
def create_and_launch_app(**kwargs):
    """Create and launch the Gradio app."""
    app = WebApp()
    app.launch(**kwargs)

# Main entry point
if __name__ == "__main__":
    create_and_launch_app(share=True)