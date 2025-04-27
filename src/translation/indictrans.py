# src/translation/indictrans.py
import torch
import os
from typing import Optional, List, Dict, Any
from pathlib import Path

from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode

class IndicTransModel:
    """
    Wrapper for IndicTrans2 model.
    """
    
    MODEL_URLS = {
        "indictrans2": "https://github.com/AI4Bharat/IndicTrans2/releases/download/v1-model-weights/indicTrans2-indic-en-0.9B.zip",
    }
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.device = self.model_manager.device
        self.model = None
        self.tokenizer = None
        self.mapping = {
            "en": "eng_Latn",
            "hi": "hin_Deva",
            "kn": "kan_Knda",
        }
        
    def load_model(self):
        """Load the IndicTrans2 model."""
        try:
            # Ensure model file is downloaded
            model_path = self.model_manager.ensure_model_downloaded(
                "indictrans2/indicTrans2-indic-en-0.9B.zip", 
                self.MODEL_URLS["indictrans2"]
            )
            
            # Extract if needed
            model_dir = model_path.parent
            if not (model_dir / "flores200_sacrebleu_tokenizer_spm.model").exists():
                import zipfile
                with zipfile.ZipFile(model_path, 'r') as zip_ref:
                    zip_ref.extractall(model_dir)
            
            # Import libraries here to avoid loading them if model not used
            from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator
            from fairseq import hub_utils
            
            # Load IndicTrans model
            ckpt_dir = str(model_dir)
            model_type = "en-indic" if self.device.type == "cpu" else "cuda-en-indic"
            
            # Load the model
            checkpoint_path = os.path.join(ckpt_dir, "model.pt")
            self.indic_to_eng_model = hub_utils.GeneratorHubInterface(
                hub_utils.from_pretrained(
                    checkpoint_path,
                    task="translation",
                    arg_overrides={"data": ckpt_dir}
                )
            )
            self.indic_to_eng_model.to(self.device)
            
            # Initialize tokenizer
            from fairseq.data import Dictionary
            self.tokenizer = Dictionary.load(os.path.join(ckpt_dir, "dict.src.txt"))
            
            print(f"Loaded IndicTrans2 model on {self.device}")
            
        except Exception as e:
            print(f"Error loading IndicTrans2 model: {e}")
            raise

    def get_lang_code(self, lang: str) -> str:
        """Convert language code to IndicTrans format."""
        return self.mapping.get(lang, "eng_Latn")
    
    def translate(self, 
                text: str, 
                src_lang: str = LanguageCode.ENGLISH,
                tgt_lang: str = LanguageCode.HINDI) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            src_lang: Source language code
            tgt_lang: Target language code
            
        Returns:
            Translated text
        """
        if self.model is None:
            self.load_model()
            
        # Special case for English-Hindi direct translation
        if (src_lang == LanguageCode.ENGLISH and tgt_lang == LanguageCode.HINDI) or \
           (src_lang == LanguageCode.HINDI and tgt_lang == LanguageCode.ENGLISH):
            # Use direct translation
            src_indic = self.get_lang_code(src_lang)
            tgt_indic = self.get_lang_code(tgt_lang)
            
            # Translate using IndicTrans
            translated = self.indic_to_eng_model.translate(
                text,
                src_lang=src_indic, 
                tgt_lang=tgt_indic
            )
            return translated
            
        # For other language pairs (e.g., Hindi-Kannada), use English as pivot
        if src_lang != LanguageCode.ENGLISH:
            # First translate to English
            src_indic = self.get_lang_code(src_lang)
            eng_indic = "eng_Latn"
            
            english_text = self.indic_to_eng_model.translate(
                text,
                src_lang=src_indic, 
                tgt_lang=eng_indic
            )
            
            # Then translate from English to target
            if tgt_lang == LanguageCode.ENGLISH:
                return english_text
                
            tgt_indic = self.get_lang_code(tgt_lang)
            return self.indic_to_eng_model.translate(
                english_text,
                src_lang=eng_indic, 
                tgt_lang=tgt_indic
            )
        else:
            # Source is English, translate to target
            src_indic = self.get_lang_code(src_lang)
            tgt_indic = self.get_lang_code(tgt_lang)
            
            return self.indic_to_eng_model.translate(
                text,
                src_lang=src_indic, 
                tgt_lang=tgt_indic
            )