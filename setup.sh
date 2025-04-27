#!/bin/bash
# setup.sh - Set up the EchoLang environment

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS system"
    PYTHON_CMD="python3"
    # Check for Apple Silicon
    if [[ $(uname -m) == 'arm64' ]]; then
        echo "Running on Apple Silicon"
    else
        echo "Warning: You're running on Intel Mac. This app is optimized for Apple Silicon."
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux system"
    PYTHON_CMD="python3"
else
    echo "Detected other OS (assuming Windows)"
    PYTHON_CMD="python"
fi

# Create Python virtual environment
echo "Creating Python virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "darwin"* ]] || [[ "$OSTYPE" == "linux-gnu"* ]]; then
    source venv/bin/activate
else
    source venv/Scripts/activate
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required packages..."
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/seamless_communication.git

# Create model directories
echo "Creating model directories..."
mkdir -p models/seamless_m4t
mkdir -p models/indictrans2
mkdir -p models/xtts_v2/speakers

# Download a small test file to verify setup
echo "Downloading test files..."
mkdir -p test_data
curl -L "https://github.com/openai/whisper/raw/main/tests/jfk.flac" -o test_data/test_en.flac

echo "Setup complete!!!"