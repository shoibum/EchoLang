# src/translation/translator.py
import torch
import logging
from typing import Optional, Dict, List
from pathlib import Path
import time
import gc

# Use relative imports for config and utils
try:
    from .. import config
    from ..utils.language import LanguageCode
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config
    from utils.language import LanguageCode

logger = logging.getLogger(__name__)

# Import transformers components
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    logger.debug("Successfully imported transformers components for IndicTrans2.")
except ImportError as e:
    logger.error("Failed to import transformers. Is it installed correctly?", exc_info=True)
    raise ImportError("Could not import transformers components.") from e

# Import IndicTransToolkit processor
try:
    from IndicTransToolkit.processor import IndicProcessor
    logger.debug("Successfully imported IndicProcessor from IndicTransToolkit.")
except ImportError as e:
    logger.error("Failed to import IndicProcessor. Is IndicTransToolkit installed via git+...? ", exc_info=True)
    IndicProcessor = None


class Translator:
    """
    Translation interface using distilled IndicTrans2 models.
    Handles En->Indic, Indic->En, and Indic->Indic directions.
    """

    def __init__(self):
        if IndicProcessor is None:
             raise RuntimeError("IndicTransToolkit is not installed or failed to import.")
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
             raise RuntimeError("transformers library is not installed or failed to import.")

        self.device = config.APP_DEVICE # Will be 'cpu' after config change
        # Sets correct dtype based on device
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"Initializing Distilled IndicTrans2 Translator. Device: {self.device}, Dtype: {self.torch_dtype}")

        self.en_indic_model_id = config.INDIC_TRANS_EN_INDIC_MODEL_ID
        self.indic_en_model_id = config.INDIC_TRANS_INDIC_EN_MODEL_ID
        self.indic_indic_model_id = config.INDIC_TRANS_INDIC_INDIC_MODEL_ID

        self.tokenizer_en_indic = None
        self.model_en_indic = None
        self.tokenizer_indic_en = None
        self.model_indic_en = None
        self.tokenizer_indic_indic = None
        self.model_indic_indic = None

        self.indic_processor = IndicProcessor(inference=True)
        self._load_models()

    def _load_model_helper(self, model_id: str, direction: str):
        """Helper to load a single model and tokenizer."""
        logger.info(f"Loading {direction} model: {model_id}")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype # Will be float32 for CPU
        ).to(self.device) # Will move to CPU
        model.eval()
        logger.info(f"Loaded {direction} model in {time.time() - start_time:.2f}s")
        return tokenizer, model

    def _load_models(self):
        """Loads all three distilled IndicTrans2 models."""
        # (Keep the _load_models method exactly as in Response #47)
        logger.info("Loading Distilled IndicTrans2 models...")
        try:
            self.tokenizer_en_indic, self.model_en_indic = self._load_model_helper(self.en_indic_model_id, "En->Indic")
            self.tokenizer_indic_en, self.model_indic_en = self._load_model_helper(self.indic_en_model_id, "Indic->En")
            self.tokenizer_indic_indic, self.model_indic_indic = self._load_model_helper(self.indic_indic_model_id, "Indic->Indic")
            logger.info("All Distilled IndicTrans2 models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Distilled IndicTrans2 models: {e}", exc_info=True)
            self.tokenizer_en_indic = self.model_en_indic = None
            self.tokenizer_indic_en = self.model_indic_en = None
            self.tokenizer_indic_indic = self.model_indic_indic = None
            raise RuntimeError(f"Failed to load one or more Distilled IndicTrans2 models") from e

    def _translate_batch(self, batch: List[str], model, tokenizer, src_lang_code: str, tgt_lang_code: str) -> List[str]:
        """Helper function to handle translation for a batch."""
        # (Keep the _translate_batch method exactly as in Response #49 - includes tokenizer context fix)
        if not model or not tokenizer:
             raise RuntimeError("Translation model or tokenizer is not loaded.")
        processed_batch = self.indic_processor.preprocess_batch(batch, src_lang=src_lang_code, tgt_lang=tgt_lang_code)
        inputs = tokenizer(processed_batch, padding="longest", truncation=True, max_length=256, return_tensors="pt", return_attention_mask=True).to(self.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, num_beams=5, max_length=256, num_return_sequences=1)
        with tokenizer.as_target_tokenizer():
            decoded_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        final_translations = self.indic_processor.postprocess_batch(decoded_tokens, lang=tgt_lang_code)
        return final_translations

    def translate(self,
                  text: str,
                  src_lang: str,
                  tgt_lang: str) -> Dict:
        # (Keep the main translate method exactly as in Response #47 - selects correct model)
        translated_text = ""
        error_message = None
        start_time = time.time()

        indic_src = config.INDIC_TRANS_LANG_CODES.get(src_lang)
        indic_tgt = config.INDIC_TRANS_LANG_CODES.get(tgt_lang)

        if not indic_src or not indic_tgt:
            error_message = f"Unsupported language code provided for IndicTrans2: {src_lang} or {tgt_lang}"
            logger.error(error_message)
        elif src_lang == tgt_lang:
            logger.warning(f"Source ({src_lang}) and Target ({tgt_lang}) languages are the same. Skipping translation.")
            translated_text = text
        else:
            model = None
            tokenizer = None
            direction = "Unknown"

            if src_lang == 'en' and tgt_lang != 'en':
                model = self.model_en_indic
                tokenizer = self.tokenizer_en_indic
                direction = "En->Indic"
            elif src_lang != 'en' and tgt_lang == 'en':
                model = self.model_indic_en
                tokenizer = self.tokenizer_indic_en
                direction = "Indic->En"
            else:
                model = self.model_indic_indic
                tokenizer = self.tokenizer_indic_indic
                direction = "Indic->Indic"

            if model and tokenizer and self.indic_processor:
                logger.info(f"Translating ({direction}) using Distilled IndicTrans2 ({self.device}): '{text[:50]}...'") # Device will now be CPU
                try:
                    translations = self._translate_batch([text], model, tokenizer, indic_src, indic_tgt)
                    translated_text = translations[0] if translations else ""

                    if not error_message:
                         logger.info(f"Distilled IndicTrans2 translation successful. Result: '{translated_text[:100]}...'")

                except Exception as e:
                    error_message = f"Distilled IndicTrans2 translation failed: {type(e).__name__} - {e}"
                    logger.error(error_message, exc_info=True)
            elif not error_message:
                 error_message = f"Appropriate IndicTrans2 model/tokenizer ({direction}) not loaded properly."
                 logger.error(error_message)

        processing_time = time.time() - start_time
        logger.info(f"Translation processing time: {processing_time:.2f}s")
        has_error = error_message is not None
        return {
            "original_text": text,
            "translated_text": translated_text if not has_error else "",
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "error": error_message
        }