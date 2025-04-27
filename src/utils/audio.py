# src/utils/audio.py
import os
import numpy as np
import soundfile as sf
import torch
import torchaudio
from typing import Optional, Tuple, Union

class AudioProcessor:
    """Audio processing utilities for EchoLang."""
    
    @staticmethod
    def load_audio(file_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
            sample_rate = target_sr
            
        return waveform, sample_rate
    
    @staticmethod
    def save_audio(waveform: torch.Tensor, file_path: str, sample_rate: int = 24000) -> str:
        """
        Save audio tensor to file.
        
        Args:
            waveform: Audio tensor
            file_path: Output file path
            sample_rate: Sample rate
            
        Returns:
            Path to saved file
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Ensure waveform is on CPU and convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
            
        # Reshape if needed
        if waveform.ndim == 1:
            waveform = waveform.reshape(1, -1)
        
        # Normalize
        if np.abs(waveform).max() > 1.0:
            waveform = waveform / np.abs(waveform).max()
            
        sf.write(file_path, waveform.T, sample_rate)
        return file_path
    
    @staticmethod
    def get_duration(file_path: str) -> float:
        """Get duration of audio file in seconds."""
        info = torchaudio.info(file_path)
        return info.num_frames / info.sample_rate