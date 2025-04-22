"""
Utility functions for audio processing.
"""

import os
import io
import tempfile
from typing import Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample it.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = 16000) -> str:
    """
    Save audio array to file.
    
    Args:
        audio: Audio array
        file_path: Output file path
        sample_rate: Sample rate
        
    Returns:
        Path to saved file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    sf.write(file_path, audio, sample_rate)
    return file_path

def convert_audio_bytes_to_wav(audio_bytes: bytes, target_sr: int = 16000) -> bytes:
    """
    Convert audio bytes (possibly in various formats) to WAV format with specified sample rate.
    
    Args:
        audio_bytes: Audio content as bytes
        target_sr: Target sample rate
        
    Returns:
        WAV audio bytes
    """
    # Save input bytes to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as temp_input:
        temp_input_path = temp_input.name
        temp_input.write(audio_bytes)
    
    try:
        # Load with pydub which can handle various formats
        audio = AudioSegment.from_file(temp_input_path)
        
        # Set channels and sample rate
        audio = audio.set_channels(1)  # Mono
        audio = audio.set_frame_rate(target_sr)
        
        # Export to WAV bytes
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        
        return buffer.getvalue()
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)

def get_audio_duration(file_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    y, sr = librosa.load(file_path, sr=None)
    return librosa.get_duration(y=y, sr=sr)