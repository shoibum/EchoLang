# src/translation/transformers_translator.py
import torch
import logging
from typing import Optional, Dict, Union, List
from pathlib import Path
import time
import gc # Garbage collector for unloading models

# Use relative imports for utils and config
from ..utils.model_utils import ModelManager
from ..utils.language import LanguageCode
try:
    from .. import config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

# Import transformers components
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    logger.debug("Successfully imported transformers components for Translation.")
except ImportError as e:
    logger.error("Failed to import transformers. Is it installed correctly?", exc_info=True)
    raise ImportError("Could not import transformers components.") from e

class TransformersModel:
    """
    Wrapper for individual Translation models using Hugging Face transformers (e.g., NLLB, MarianMT).
    Loads a specific model based on configuration. FORCES CPU execution.
    """

    def __init__(self, model_config: Dict, model_manager: Optional[ModelManager] = None):
        """
        Initializes the wrapper for a specific translation model config.

        Args:
            model_config (Dict): The configuration dictionary for the specific model
                                 (e.g., an entry from MARIAN_CONFIG or NLLB_CONFIG).
                                 Must contain 'hf_id' and 'model_type'.
            model_manager (Optional[ModelManager]): Model manager instance.
        """
        self.model_manager = model_manager or ModelManager()
        # --- CHANGE: Force CPU for Translation Models ---
        self.device = "cpu"
        self.torch_dtype = torch.float32 # Use float32 for CPU
        # --- End Change ---

        self.hf_id = model_config.get("hf_id")
        self.model_type = model_config.get("model_type") # Get type from config

        if not self.hf_id:
            raise ValueError("Model config must contain an 'hf_id'.")
        if not self.model_type:
            raise ValueError("Model config must contain a 'model_type' ('marian' or 'nllb').")

        self.model: Optional[AutoModelForSeq2SeqLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        # --- Updated Log ---
        logger.info(f"Initializing TransformersModel wrapper for: {self.hf_id} (Type: {self.model_type})")
        logger.info(f"FORCING CPU execution for this translation model (device={self.device}, dtype={self.torch_dtype})")
        # --- End Update ---

    def load_model(self):
        """Load the specified model and tokenizer onto the CPU."""
        if self.model is not None and self.tokenizer is not None:
            logger.debug(f"Model {self.hf_id} already loaded.")
            return

        logger.info(f"Loading translation model: {self.hf_id} onto CPU.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
            # --- CHANGE: Ensure float32 dtype for CPU loading ---
            load_dtype = torch.float32
            logger.info(f"Loading model {self.hf_id} with dtype: {load_dtype}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.hf_id,
                torch_dtype=load_dtype
            )
            # --- CHANGE: Ensure model is explicitly on CPU ---
            self.model.to(self.device) # self.device is now "cpu"
            self.model.eval() # Set to evaluation mode
            logger.info(f"Successfully loaded model {self.hf_id} and tokenizer onto {self.device}.")
            # --- End Changes ---

        except Exception as e:
            logger.error(f"Error loading translation model {self.hf_id}: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load model {self.hf_id}") from e

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text using the loaded model (on CPU).
        Handles model-specific requirements (e.g., NLLB language codes).

        Args:
            text (str): Text to translate.
            src_lang (str): Source language code (internal, e.g., 'en').
            tgt_lang (str): Target language code (internal, e.g., 'hi').

        Returns:
            str: Translated text or error message starting with "ERROR:".
        """
        if self.model is None or self.tokenizer is None:
            logger.error(f"Model {self.hf_id} is not loaded. Cannot translate.")
            return f"ERROR: Model {self.hf_id} not loaded"

        logger.info(f"Starting translation with {self.hf_id} (CPU): '{text[:50]}...' from {src_lang} to {tgt_lang}")

        try:
            start_time = time.time()

            # --- Model-Specific Handling ---
            generate_args = {} # Default empty args
            if self.model_type == "nllb":
                src_code = LanguageCode.code_to_nllb_code(src_lang)
                tgt_code = LanguageCode.code_to_nllb_code(tgt_lang)
                if not src_code or not tgt_code:
                    logger.error(f"NLLB: Invalid language codes {src_lang}->{src_code} or {tgt_lang}->{tgt_code}")
                    return "ERROR: Invalid language code for NLLB model"

                # NLLB requires setting src_lang and forcing target BOS token
                if hasattr(self.tokenizer, 'src_lang'):
                    self.tokenizer.src_lang = src_code
                else:
                    logger.warning(f"Tokenizer for {self.hf_id} does not have src_lang attribute. Proceeding without setting it.")

                forced_bos_token_id = None
                if hasattr(self.tokenizer, 'lang_code_to_id') and tgt_code in self.tokenizer.lang_code_to_id:
                     forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_code]
                     logger.debug(f"NLLB: Using src_code={src_code}, tgt_code={tgt_code}, forced_bos_id={forced_bos_token_id}")
                     generate_args = {"forced_bos_token_id": forced_bos_token_id}
                elif hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                     try:
                          bos_token = tgt_code
                          forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(bos_token)
                          if forced_bos_token_id != self.tokenizer.unk_token_id:
                              logger.debug(f"NLLB: Using src_code={src_code}, tgt_code={tgt_code}, forced_bos_id={forced_bos_token_id} (via convert_tokens_to_ids)")
                              generate_args = {"forced_bos_token_id": forced_bos_token_id}
                          else:
                              logger.warning(f"Could not get valid BOS token ID for '{tgt_code}' using convert_tokens_to_ids. Proceeding without forced_bos_token_id.")
                     except Exception as conv_err:
                          logger.warning(f"Error getting BOS token ID for '{tgt_code}': {conv_err}. Proceeding without forced_bos_token_id.")
                else:
                     logger.warning(f"Could not find target language code '{tgt_code}' in tokenizer attributes for {self.hf_id}. Proceeding without forced_bos_token_id.")

            elif self.model_type == "marian":
                logger.debug(f"MarianMT ({self.hf_id}): No special language codes needed for generation.")
            # -----------------------------

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            # --- CHANGE: Ensure inputs are on CPU ---
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # self.device is "cpu"
            # --- End Change ---

            # --- Use greedy search (num_beams=1, do_sample=False) ---
            logger.debug(f"Generating translation using greedy search (CPU) (num_beams=1, do_sample=False, max_length={512})")
            translated_tokens = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=1,
                do_sample=False,
                early_stopping=False,
                **generate_args
            )
            # --- End of Change ---

            # Decode the generated tokens
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            end_time = time.time()
            logger.info(f"{self.hf_id} translation successful (CPU) in {end_time - start_time:.2f}s. Result: '{translated_text[:100]}...'")
            return translated_text

        except Exception as e:
            logger.error(f"Error during {self.hf_id} translation from {src_lang} to {tgt_lang} (CPU): {e}", exc_info=True)
            error_detail = str(e).split('\n')[0][:200]
            return f"ERROR: Translation failed ({error_detail})"

    def unload_model(self):
        """Unload model and tokenizer to free memory."""
        if self.model is not None:
            logger.info(f"Unloading model: {self.hf_id}")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            # No need for cuda/mps empty_cache when using CPU
            gc.collect() # Trigger garbage collection
            logger.info(f"Model {self.hf_id} unloaded.")

