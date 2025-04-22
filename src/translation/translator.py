"""
Translation module for multilingual text translation.
"""

from typing import Dict
import torch
from transformers import pipeline
from src.config import LANGUAGES

# ──────────────────────────────────────────────────
# Map each supported pair to a real HF model ID.
# ──────────────────────────────────────────────────
PAIR2MODEL: Dict[tuple[str,str], str] = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("en", "kn"): "ai4bharat/indictrans2-en-indic-1B",
    ("kn", "en"): "ai4bharat/indictrans2-indic-en-dist-200M",
    # hi↔kn will fall back via English
}


class Translator:
    """
    Translator class using Hugging Face Transformers for
    multilingual text translation with automatic fallback.
    """
    
    def __init__(self):
        # HF pipeline expects device index: 0 for GPU, -1 for CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipelines: Dict[str, pipeline] = {}
        print(f"Using device: {'cuda' if self.device==0 else 'cpu'}")
    
    def _make_key(self, src: str, tgt: str) -> str:
        return f"{src}-{tgt}"
    
    def load_model(self, src: str, tgt: str) -> None:
        """Ensure a pipeline exists for src→tgt, or set up fallback via English."""
        if src == tgt:
            return
        if src not in LANGUAGES or tgt not in LANGUAGES:
            raise ValueError(f"Unsupported language: {src} or {tgt}")
        
        key = self._make_key(src, tgt)
        if key in self.pipelines:
            return
        
        model_id = PAIR2MODEL.get((src, tgt))
        if model_id:
            print(f"Loading model {src}→{tgt}: {model_id}")
            self.pipelines[key] = pipeline(
                "translation",
                model=model_id,
                device=self.device,
                trust_remote_code=True,   # ← allow custom code for IndicTrans2
                max_length=512,
            )
            return
        
        # No direct model → fallback via English
        print(f"No direct model for {src}→{tgt}, falling back via English")
        if src != "en":
            self.load_model(src, "en")
        if tgt != "en":
            self.load_model("en", tgt)
    
    def translate(self, text: str, src: str, tgt: str) -> str:
        """
        Translate *text* from *src* language to *tgt* language.
        Falls back via English for unsupported direct pairs.
        """
        if src == tgt:
            return text
        
        self.load_model(src, tgt)
        
        if (src, tgt) in PAIR2MODEL:
            key = self._make_key(src, tgt)
            out = self.pipelines[key](text)
            return out[0]["translation_text"]
        
        # Fallback via English
        if src != "en":
            text = self.translate(text, src, "en")
        if tgt != "en":
            text = self.translate(text, "en", tgt)
        return text
