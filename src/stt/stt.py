"""
Speech-to-Text module for multilingual speech recognition.
Falls back to CPU for Whisper on MPS, and uses a Kannada‑native
Wav2Vec2 model when the user requests Kannada (kn).
"""

import os
import tempfile
import whisper
import torch
import soundfile as sf
import librosa
from pathlib import Path
from typing import Dict, Optional, Union, BinaryIO

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from src.config import WHISPER_MODEL_SIZE, MODELS_DIR, LANGUAGES


class SpeechToText:
    """
    Whisper wrapper with MPS→CPU fallback, plus a special path
    for Kannada via Wav2Vec2.
    """

    KANNADA_MODEL = "addy88/wav2vec2-kannada-stt"  # or "amoghsgopadi/wav2vec2-large-xlsr-kn"

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        self.model_size = model_size
        self.model = None
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"[STT] Preferred device: {self.device}")

    def load_model(self):
        """Load Whisper model, falling back to CPU on MPS sparse-op errors."""
        if self.model is not None:
            return
        try:
            print(f"[STT] Loading Whisper '{self.model_size}' on {self.device}…")
            self.model = whisper.load_model(self.model_size, device=self.device)
        except NotImplementedError as e:
            print(f"[STT] MPS error:\n  {e}\n[STT] Falling back to CPU…")
            self.device = "cpu"
            self.model = whisper.load_model(self.model_size, device="cpu")
        print(f"[STT] Whisper loaded on {self.device}!")

    def transcribe(
        self,
        audio_file: Union[str, Path, BinaryIO],
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe speech → text. If language=='kn', use Wav2Vec2‑Kannada;
        otherwise use Whisper (auto‑fallback).
        """
        # 1) Get a filesystem path
        temp_created = False
        if not isinstance(audio_file, (str, Path)):
            tf = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tf.write(audio_file.read())
            tf.close()
            audio_path = tf.name
            temp_created = True
        else:
            audio_path = str(audio_file)

        try:
            # 2) Kannada special branch
            if language == "kn":
                return self._transcribe_kannada(audio_path)

            # 3) Default: Whisper
            self.load_model()
            opts = {}
            if language:
                opts["language"] = language
            result = self.model.transcribe(audio_path, **opts)
            return {
                "text": result["text"].strip(),
                "language": result.get("language", "unknown"),
                "segments": result.get("segments", []),
            }
        finally:
            if temp_created and os.path.exists(audio_path):
                os.unlink(audio_path)

    def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None
    ) -> Dict:
        """Entry‑point for raw bytes."""
        import io
        return self.transcribe(io.BytesIO(audio_bytes), language)

    # ────────────────────────────────────────────────────────────
    # Kannada via Wav2Vec2
    # ────────────────────────────────────────────────────────────
    def _transcribe_kannada(self, wav_path: str) -> Dict:
        """
        Use a Kannada‑fine‑tuned Wav2Vec2 model that outputs
        Devanagari natively.
        """
        # 📥 load processor & model
        processor = Wav2Vec2Processor.from_pretrained(self.KANNADA_MODEL)
        model     = Wav2Vec2ForCTC.from_pretrained(self.KANNADA_MODEL).to(self.device)

        # 📂 read audio & resample to 16k
        speech, sr = sf.read(wav_path)
        if sr != 16000:
            speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
            sr = 16000

        # 🧩 tokenize & infer
        inputs = processor(speech, sampling_rate=sr, return_tensors="pt", padding=True)
        logits = model(inputs.input_values.to(self.device)).logits
        pred_ids = torch.argmax(logits, dim=-1)

        # 🔤 decode → native Devanagari
        text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        return {"text": text.strip(), "language": "kn", "segments": []}
