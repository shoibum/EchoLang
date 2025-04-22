"""
Text‑to‑Speech module for multilingual speech synthesis.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager

from src.config import TTS_MODELS, LANGUAGES


class TextToSpeech:
    """Coqui‑TTS wrapper that supports English, Hindi, and Kannada."""

    def __init__(self) -> None:
        # download helper (no local models.json so it pulls from HuggingFace)
        self.model_manager = ModelManager(models_file=None)

        # language‑specific Synthesizer objects live here
        self.synthesizers: Dict[str, Synthesizer] = {}

        # choose compute device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using device: {self.device}")

    # ──────────────────────────────────────────────────────────
    # Load and cache a TTS model for the given language
    # ──────────────────────────────────────────────────────────
    def load_model(self, language: str) -> None:
        """Download (if needed) and initialise the TTS model for *language*."""

        if language not in LANGUAGES:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported languages are: {', '.join(LANGUAGES.keys())}"
            )

        # already loaded
        if language in self.synthesizers:
            return

        print(f"Loading TTS model for {LANGUAGES[language]}…")
        model_name = TTS_MODELS[language]

        # ── 1. download / locate model files ───────────────────────────────
        paths = self.model_manager.download_model(model_name)
        if paths is None:
            raise RuntimeError(f"Could not download model “{model_name}”")

        # paths[0] = checkpoint (.pth)   • paths[1] = config (.json/.yaml)
        tts_checkpoint_path = paths[0]
        tts_config_path     = paths[1]

        # vocoder assets – layout differs by TTS version
        vocoder_checkpoint_path = vocoder_config_path = None
        if len(paths) == 4:        # 0.20–0.21.x → [ckpt, cfg, VOC_CFG, VOC_CKPT]
            vocoder_config_path     = paths[2]
            vocoder_checkpoint_path = paths[3]
        elif len(paths) >= 5:      # very old → [ckpt, cfg, SPK_CFG, VOC_CFG, VOC_CKPT]
            vocoder_config_path     = paths[3]
            vocoder_checkpoint_path = paths[4]
        # len == 3 → vocoder is packaged inside the TTS model (≥0.22)

        # ── 2. create Synthesizer ─────────────────────────────────────────
        self.synthesizers[language] = Synthesizer(
            tts_checkpoint=tts_checkpoint_path,
            tts_config_path=tts_config_path,
            vocoder_checkpoint=vocoder_checkpoint_path,
            vocoder_config=vocoder_config_path,
            use_cuda=self.device == "cuda",
        )

        print(f"TTS model for {LANGUAGES[language]} loaded successfully!")

    # ──────────────────────────────────────────────────────────
    # Synthesize speech
    # ──────────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        language: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Union[str, bytes]:
        """
        Convert *text* to speech in *language*.
        Returns either the file path (if *output_path* was provided) or
        raw WAV bytes (if the method created a temp file).
        """

        # Ensure the model for this language is ready
        self.load_model(language)

        # Prepare output destination
        temp_created = False
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = temp_file.name
            temp_file.close()
            temp_created = True

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Generate audio
        wav = self.synthesizers[language].tts(text)
        self.synthesizers[language].save_wav(wav, output_path)

        # Return bytes and delete temp, or just return the path
        if temp_created:
            with open(output_path, "rb") as f:
                data = f.read()
            os.unlink(output_path)
            return data
        else:
            return output_path
