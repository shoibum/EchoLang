#!/bin/bash
# setup.sh - Set up the EchoLang environment (Updated for Transformers)
# IMPORTANT: On Windows, run this script using Git Bash or WSL.
# Prerequisites: bash, python3 (or python mapped to python3), pip, git, curl

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Prerequisite Checks (Optional but Recommended) ---
command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || { echo >&2 "‼️ Python (python3 or python) is required but not found. Aborting."; exit 1; }
command -v git >/dev/null 2>&1 || { echo >&2 "‼️ git is required but not found. Aborting."; exit 1; }
command -v curl >/dev/null 2>&1 || { echo >&2 "‼️ curl is required but not found. Aborting."; exit 1; }
echo "✅ Prerequisites found."

# --- Setup Steps ---

# Detect OS & Set Python Command
PYTHON_CMD=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍏 Detected macOS system"
    PYTHON_CMD="python3"
    # Check for Apple Silicon
    if [[ $(uname -m) == 'arm64' ]]; then
        echo "   Running on Apple Silicon (arm64)"
    else
        echo "   ⚠️ Running on Intel Mac. Ensure PyTorch is installed correctly."
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Detected Linux system"
    PYTHON_CMD="python3"
else
    # Assuming Windows (requires Git Bash/WSL) or other Unix-like
    echo "💻 Detected Windows/Other system (assuming python points to Python 3)"
    PYTHON_CMD="python" # User must ensure 'python' is Python 3 in their environment
fi

# Verify the selected Python command works and find pip
if ! command -v $PYTHON_CMD >/dev/null 2>&1; then
     echo >&2 "‼️ Selected Python command '$PYTHON_CMD' not found. Aborting."
     exit 1
fi
PIP_CMD="$PYTHON_CMD -m pip"
echo "   Using Python command: $PYTHON_CMD"
echo "   Using Pip command: $PIP_CMD"


# Create Python virtual environment
VENV_DIR="venv"
echo "🐍 Creating Python virtual environment in '$VENV_DIR'..."
$PYTHON_CMD -m venv $VENV_DIR

# Activate virtual environment (for script execution only)
echo "   Activating virtual environment for subsequent commands..."
source "$VENV_DIR/bin/activate" || source "$VENV_DIR/Scripts/activate" || { echo >&2 "‼️ Failed to activate venv."; exit 1; }
echo "   Virtual environment activated for script."

# Install required packages
echo "📦 Installing required packages from requirements.txt..."
$PIP_CMD install -r requirements.txt

# Create necessary directories (only XTTS now)
echo "📁 Creating model directories (for XTTS)..."
mkdir -p models/xtts_v2/speakers # Ensure speakers dir exists

# Download a small test file (optional)
echo "🌐 Downloading test audio file (optional)..."
mkdir -p test_data
curl -L "https://github.com/openai/whisper/raw/main/tests/jfk.flac" -o test_data/test_en.flac


# --- Completion ---
echo ""
echo "✅ Setup complete!"
echo "   To activate the virtual environment in your terminal, run:"
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "      source $VENV_DIR/bin/activate"
else
    # Provide instructions for common Windows shells
    echo "      (Git Bash/WSL): source $VENV_DIR/Scripts/activate"
    echo "      (CMD)         : %cd%\\$VENV_DIR\\Scripts\\activate.bat"
    echo "      (PowerShell)  : .\\$VENV_DIR\\Scripts\\Activate.ps1"
fi
echo ""
echo "➡️ Next step: Run 'python main.py'"
echo ""