#!/bin/bash
# reset_models.sh - Reset model files (Updated for Transformers+XTTS)

echo "WARNING: This will remove downloaded XTTS model files from the './models' directory."
echo "         It will NOT clear the main Hugging Face cache (~/.cache/huggingface/hub) used by Transformers (Whisper, NLLB)."
echo "         To clear the HF cache, delete the relevant subfolders inside ~/.cache/huggingface/hub/models--* manually."
read -p "Do you want to remove the local XTTS model files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Removing local XTTS model files..."
    rm -rf models/xtts_v2/*

    # Keep the directory structure
    mkdir -p models/xtts_v2/speakers

    echo "Local XTTS model files removed. They will be redownloaded on next TTS usage."
else
    echo "Operation cancelled."
fi