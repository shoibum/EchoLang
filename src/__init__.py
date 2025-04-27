# src/__init__.py
"""EchoLang - Multilingual Speech-to-Text-to-Speech Pipeline."""

import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Package version
__version__ = "1.0.0"