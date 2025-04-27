"""
Seamless M4T-Large ASR wrapper (GPU-safe on Apple Silicon).
"""

import torch, soundfile as sf
from pathlib import Path
from transformers import AutoProcessor, SeamlessM4TForSpeechToText

MODEL_ID = "facebook/seamless-m4t-v2-large"   # or "facebook/hf-seamless-m4t-large"
DEVICE   = "mps" if torch.backends.mps.is_available() else \
           "cuda" if torch.cuda.is_available() else "cpu"

print(f"[M4T] loading {MODEL_ID} on {DEVICE}…")
processor = AutoProcessor.from_pretrained(
    MODEL_ID, trust_remote_code=True, cache_dir=str(Path.home()/".cache/hf")
)
model = SeamlessM4TForSpeechToText.from_pretrained(
    MODEL_ID, trust_remote_code=True, cache_dir=str(Path.home()/".cache/hf")
).to(DEVICE)
print("[M4T] ready!")

def transcribe(wav_path: str, tgt_lang: str | None = "en") -> str:
    """Return plain text in *tgt_lang* (ISO-639-1)."""
    audio, sr = sf.read(wav_path)
    inputs    = processor(audios=[audio], sampling_rate=sr, return_tensors="pt").to(DEVICE)
    gen_ids   = model.generate(**inputs, tgt_lang=tgt_lang or "en")
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
