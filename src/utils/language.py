# src/utils/language.py
from enum import Enum
from typing import Dict, List, Optional

try:
    from .. import config # Use central language dicts
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

class LanguageCode(str, Enum):
    """Language codes used internally in EchoLang."""
    ENGLISH = "en"
    HINDI = "hi"
    KANNADA = "kn"

    @classmethod
    def list(cls) -> List[str]:
        """Get list of all supported language codes."""
        return [lang.value for lang in cls]

    @classmethod
    def list_human_readable(cls) -> List[str]:
        """Get list of human-readable names."""
        return list(config.LANGUAGES.values())

    @classmethod
    def get_choices_for_gradio(cls) -> List[tuple[str, str]]:
         """Get list of (Name, Code) tuples for Gradio Dropdown."""
         return [(name, code) for code, name in config.LANGUAGES.items()]

    @classmethod
    def name_to_code(cls, name: str) -> Optional[str]:
        """Convert language name to language code."""
        name_lower = name.lower()
        for code, lang_name in config.LANGUAGES.items():
            if lang_name.lower() == name_lower:
                return code
        return None # Return None if not found

    @classmethod
    def code_to_name(cls, code: str) -> str:
        """Convert language code to human-readable name."""
        return config.LANGUAGES.get(code.lower(), "Unknown")

    @classmethod
    def code_to_xtts_locale(cls, code: str) -> str:
        """Convert language code to XTTS locale (often same as code)."""
        # XTTS v2 uses standard codes directly for supported languages
        code_lower = code.lower()
        if code_lower in config.LANGUAGES:
            return code_lower
        return "en" # Default fallback

    @classmethod
    def code_to_whisper_lang(cls, code: str) -> str:
        """Convert language code to Whisper language name (full name lowercase)."""
        # Whisper generally expects the full language name in lowercase
        # or can auto-detect. We provide the name for clarity.
        return config.LANGUAGES.get(code.lower(), "english").lower()

    @classmethod
    def code_to_nllb_code(cls, code: str) -> Optional[str]:
        """Convert language code to NLLB format (e.g., 'eng_Latn')."""
        return config.NLLB_LANG_CODES.get(code.lower())

    # Removed code_to_m4t_code as M4T is no longer used