#!/usr/bin/env bash
#
# EchoLang setup – Seamless M4T-Large  +  IndicTrans-2  +  XTTS-v2
# • Re-uses current venv if $VIRTUAL_ENV is set, else creates .venv/
# • Installs Torch 2.6.0 CPU/MPS wheels (GPU-safe on Apple-Silicon)
# • Pulls Transformers ≥ 4.40 (+ trust_remote_code), SentencePiece, Accelerate
# • Installs TTS 0.22.0 (XTTS-v2), Gradio 4.44.0 and audio libs
# • Downloads all checkpoints on first run (~5 GB total)

set -e

# ──────────────────────────────────────────────────────────
# 1  Activate / create virtual-env
# ──────────────────────────────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -d .venv ]]; then
    echo "🔄  Re-using existing .venv/"
  else
    echo "🐍  Creating .venv/ (Python 3.11)…"
    python3.11 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  echo "🔗  Using active venv at $VIRTUAL_ENV"
fi

# ──────────────────────────────────────────────────────────
# 2  Remove any old Whisper wheel (optional cleanup)
# ──────────────────────────────────────────────────────────
pip uninstall -y openai-whisper whisper || true

# ──────────────────────────────────────────────────────────
# 3  Install / upgrade dependencies
# ──────────────────────────────────────────────────────────
echo "⬇️  Installing Python packages…"
pip install -U pip wheel

# Torch 2.6.0 + torchaudio 2.6.0 + torchvision (Metal backend works)
pip install \
  torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cpu

# Core libraries
pip install \
  "transformers[torch]>=4.40" \
  sentencepiece accelerate \
  TTS==0.22.0 \
  gradio==4.44.0 librosa soundfile pydub

# ──────────────────────────────────────────────────────────
# 4  Download checkpoints (first run only)
# ──────────────────────────────────────────────────────────
echo "🎙  Triggering model downloads …"
python - <<'PY'
from pathlib import Path
from src.pipeline import Pipeline

p = Pipeline()                                  # loads Seamless M4T, IndicTrans-2, XTTS-v2
p.speech_to_text(b'\0'*32000)                   # ASR   (Seamless M4T-Large)
p.translate_text("test", "en", "hi")            # MT   (IndicTrans-2)
p.text_to_speech("test", "en")                  # TTS  (XTTS-v2)

print("✅  All models cached in", Path("models/").resolve())
PY

echo "✅  Setup complete."
echo "👉 Launch the UI with:"
echo "   source .venv/bin/activate && python -m src.web.app --share"