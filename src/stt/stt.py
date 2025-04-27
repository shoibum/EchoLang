"""
STT layer – now delegates to Seamless M4T-Large.
"""

import os, tempfile, io
from pathlib import Path
from typing import Dict, Optional, Union, BinaryIO
from src.stt import m4t_asr          # ← new wrapper
from src.config import LANGUAGES

class SpeechToText:
    """
    Thin adapter around Seamless M4T-Large ASR.
    """

    device = m4t_asr.DEVICE  # exposed for diagnostics

    # ------------------------------------------------------------------
    def _path_from_any(self, audio: Union[str, Path, BinaryIO, bytes]) -> tuple[str, bool]:
        """Return (wav_path, temp_created?)."""
        if isinstance(audio, (bytes, bytearray)):
            audio = io.BytesIO(audio)
        if not isinstance(audio, (str, Path)):
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tf.write(audio.read())
            tf.close()
            return tf.name, True
        return str(audio), False

    # ------------------------------------------------------------------
    def transcribe(self, audio: Union[str, Path, BinaryIO, bytes],
                   language: Optional[str] = None) -> Dict:
        wav_path, tmp = self._path_from_any(audio)
        try:
            text = m4t_asr.transcribe(wav_path, tgt_lang=language)
            return {"text": text.strip(),
                    "language": language or "unknown",
                    "segments": []}
        finally:
            if tmp and os.path.exists(wav_path):
                os.unlink(wav_path)

    # legacy convenience
    def transcribe_audio_bytes(self, audio_bytes: bytes,
                               language: Optional[str] = None) -> Dict:
        return self.transcribe(io.BytesIO(audio_bytes), language)