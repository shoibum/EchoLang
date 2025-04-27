"""
Multilingual Text-to-Speech using Coqui XTTS-v2.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union
import tempfile, os, torch
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from src.config import TTS_MODEL_ID, LANGUAGES, MODELS_DIR


class TextToSpeech:
    def __init__(self) -> None:
        self.device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        self._manager = ModelManager(models_file=None, progress_bar=False)
        self.synth: Synthesizer | None = None

    # ────────────────────────────────────────────────────
    def _load(self) -> None:
        if self.synth is not None:
            return
        ckpt, cfg, *_ = self._manager.download_model(
            TTS_MODEL_ID, target_dir=str(MODELS_DIR / "tts")
        )
        self.synth = Synthesizer(
            tts_checkpoint=ckpt,
            tts_config_path=cfg,
            use_cuda=self.device == "cuda",
        )

    # ────────────────────────────────────────────────────
    def synthesize(
        self,
        text: str,
        language: str,
        out_path: Optional[Union[str, Path]] = None,
    ) -> Union[bytes, str]:
        assert language in LANGUAGES, f"Unsupported language: {language}"
        self._load()

        tmp_created = False
        if out_path is None:
            tmp_created = True
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            out_path = f.name
            f.close()

        wav = self.synth.tts(text, language_lang=language)
        self.synth.save_wav(wav, out_path)

        if tmp_created:
            data = Path(out_path).read_bytes()
            os.unlink(out_path)
            return data
        return str(out_path)