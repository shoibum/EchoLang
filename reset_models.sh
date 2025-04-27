#!/usr/bin/env bash
# Remove ALL locally-cached model weights:
# • Seamless M4T  • IndicTrans-2  • XTTS-v2
# • legacy Whisper / TTS caches   • local ./models folder

set -e
echo "🧹  Removing cached model directories …"

# Hugging Face hub (Seamless M4T, IndicTrans-2, XTTS-v2)
rm -rf ~/.cache/huggingface ~/.cache/hf 2>/dev/null || true
rm -rf ~/Library/Caches/huggingface     2>/dev/null || true

# Old Whisper + Coqui-TTS caches (if they still exist)
rm -rf ~/.cache/whisper ~/Library/Caches/whisper 2>/dev/null || true
rm -rf ~/.cache/tts     ~/.local/share/tts       2>/dev/null || true

# Project-local model cache
rm -rf "$(dirname "$0")/models" 2>/dev/null || true

echo "✅  All model caches removed."
echo "ℹ️  Next launch will download fresh checkpoints."
