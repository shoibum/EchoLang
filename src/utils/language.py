# src/utils/language.py
from enum import Enum
from typing import Dict, List, Optional

class LanguageCode(str, Enum):
    """Language codes used in EchoLang."""
    ENGLISH = "en"
    HINDI = "hi"
    KANNADA = "kn"
    
    @classmethod
    def list(cls) -> List[str]:
        """Get list of all supported language codes."""
        return [lang.value for lang in cls]
        
    @classmethod
    def name_to_code(cls, name: str) -> str:
        """Convert language name to language code."""
        name_lower = name.lower()
        mapping = {
            "english": cls.ENGLISH,
            "hindi": cls.HINDI,
            "kannada": cls.KANNADA,
        }
        return mapping.get(name_lower, cls.ENGLISH)
    
    @classmethod
    def code_to_name(cls, code: str) -> str:
        """Convert language code to human-readable name."""
        code_lower = code.lower()
        mapping = {
            cls.ENGLISH: "English",
            cls.HINDI: "Hindi",
            cls.KANNADA: "Kannada",
        }
        return mapping.get(code_lower, "Unknown")
        
    @classmethod
    def code_to_xtts_locale(cls, code: str) -> str:
        """Convert language code to XTTS locale."""
        code_lower = code.lower()
        mapping = {
            cls.ENGLISH: "en",
            cls.HINDI: "hi",
            cls.KANNADA: "kn",
        }
        return mapping.get(code_lower, "en")
        
    @classmethod
    def code_to_m4t_code(cls, code: str) -> str:
        """Convert language code to Seamless M4T code."""
        code_lower = code.lower()
        mapping = {
            cls.ENGLISH: "eng",
            cls.HINDI: "hin",
            cls.KANNADA: "kan",
        }
        return mapping.get(code_lower, "eng")