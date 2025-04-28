#!/bin/bash
# reset_models.sh - Reset model files (Updated for Transformers+XTTS)

echo "WARNING: This will remove downloaded XTTS model files from the './models' directory."
echo "         It will NOT clear the main Hugging Face cache (~/.cache/huggingface/hub) used by Transformers (Whisper, NLLB)."
echo "         It will NOT clear the locally converted CTranslate2 models (FasterWhisper STT) in ./models/"
echo "         To clear the HF cache, delete the relevant subfolders inside ~/.cache/huggingface/hub/models--* manually."
echo "         To clear converted STT models, delete models/*-ct2 directories manually."

read -p "Do you want to remove the local XTTS model files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing local XTTS model files..."
    # Only remove XTTS model files, keep speakers and CTranslate2 models
    rm -f models/xtts_v2/model.pth
    rm -f models/xtts_v2/config.json
    rm -f models/xtts_v2/vocab.json

    # Keep the directory structure
    mkdir -p models/xtts_v2/speakers

    echo "Local XTTS model files removed. They will be redownloaded on next TTS usage."
else
    echo "Operation cancelled."
fi