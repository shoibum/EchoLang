"""
Text translation via IndicTrans-2.
"""

from __future__ import annotations
from typing import Dict, Tuple
import torch
from transformers import pipeline
from src.config import PAIR2MODEL, LANGUAGES, MODELS_DIR


class Translator:
    def __init__(self) -> None:
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipes: Dict[Tuple[str, str], pipeline] = {}

    # ────────────────────────────────────────────────────
    def _ensure(self, src: str, tgt: str) -> None:
        if (src, tgt) in self.pipes or src == tgt:
            return

        model_id = PAIR2MODEL.get((src, tgt))
        if not model_id:
            # fall back via English
            if src != "en":
                self._ensure(src, "en")
            if tgt != "en":
                self._ensure("en", tgt)
            return

        self.pipes[(src, tgt)] = pipeline(
            "translation",
            model=model_id,
            device=self.device,
            cache_dir=str(MODELS_DIR / "hf"),
            max_length=512,
            trust_remote_code=True,
        )

    # ────────────────────────────────────────────────────
    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt:
            return text
        self._ensure(src, tgt)

        if (src, tgt) in self.pipes:
            out = self.pipes[(src, tgt)](text, clean_up_tokenization_spaces=True)
            return out[0]["translation_text"]

        # two-step via English
        if src != "en":
            text = self.translate(text, src, "en")
        if tgt != "en":
            text = self.translate(text, "en", tgt)
        return text