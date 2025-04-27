# src/translation/translator.py
from typing import Optional, Dict, Union
from pathlib import Path

from .indictrans import IndicTransModel
from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode

class Translator:
    """
    Translation interface for EchoLang.
    """
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.translation_model = IndicTransModel(self.model_manager)
        
    def translate(self, 
                text: str, 
                src_lang: str = LanguageCode.ENGLISH,
                tgt_lang: str = LanguageCode.HINDI) -> Dict:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            Dictionary with translation results
        """
        translated_text = self.translation_model.translate(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang
        )
        
        return {
            "original_text": text,
            "translated_text": translated_text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }