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
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    logger.debug("Successfully imported transformers components for Translation.")
except ImportError as e:
    logger.error("Failed to import transformers. Is it installed correctly?", exc_info=True)
    raise ImportError("Could not import transformers components.") from e

class TransformersModel:
    """
    Wrapper for individual Translation models using Hugging Face transformers (e.g., NLLB, MarianMT).
    Loads a specific model based on configuration.
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
        self.device = config.APP_DEVICE # Use global device setting (mps/cuda/cpu)
        self.torch_dtype = config.APP_TORCH_DTYPE # Use global dtype

        self.hf_id = model_config.get("hf_id")
        self.model_type = model_config.get("model_type") # Get type from config

        if not self.hf_id:
            raise ValueError("Model config must contain an 'hf_id'.")
        if not self.model_type:
            raise ValueError("Model config must contain a 'model_type' ('marian' or 'nllb').")


        self.model = None
        self.tokenizer = None
        logger.info(f"Initializing TransformersModel wrapper for: {self.hf_id} (Type: {self.model_type})")
        logger.info(f"Target device: {self.device}, dtype: {self.torch_dtype}")

    def load_model(self):
        """Load the specified model and tokenizer."""
        if self.model is not None and self.tokenizer is not None:
            logger.debug(f"Model {self.hf_id} already loaded.")
            return

        logger.info(f"Loading translation model: {self.hf_id}")
        # Ensure MPS fallback is enabled if needed (primarily for NLLB on MPS)
        if self.device == "mps":
             logger.info("Requesting MPS device. Ensure PYTORCH_ENABLE_MPS_FALLBACK=1 is set if needed.")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id)
            # Load with appropriate dtype for the device
            # Use float32 for CPU regardless of original dtype for broader compatibility
            load_dtype = self.torch_dtype if self.device != "cpu" else torch.float32
            logger.info(f"Loading model with dtype: {load_dtype}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.hf_id,
                torch_dtype=load_dtype
            )
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            logger.info(f"Successfully loaded model {self.hf_id} and tokenizer onto {self.device}.")

        except Exception as e:
            logger.error(f"Error loading translation model {self.hf_id}: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            raise RuntimeError(f"Failed to load model {self.hf_id}") from e

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text using the loaded model.
        Handles model-specific requirements (e.g., NLLB language codes).

        Args:
            text (str): Text to translate.
            src_lang (str): Source language code (internal, e.g., 'en').
            tgt_lang (str): Target language code (internal, e.g., 'hi').

        Returns:
            str: Translated text or error message.
        """
        if self.model is None or self.tokenizer is None:
            logger.error(f"Model {self.hf_id} is not loaded. Cannot translate.")
            return f"ERROR: Model {self.hf_id} not loaded"

        logger.info(f"Starting translation with {self.hf_id}: '{text[:50]}...' from {src_lang} to {tgt_lang}")

        try:
            start_time = time.time()

            # --- Model-Specific Handling ---
            if self.model_type == "nllb":
                src_code = LanguageCode.code_to_nllb_code(src_lang)
                tgt_code = LanguageCode.code_to_nllb_code(tgt_lang)
                if not src_code or not tgt_code:
                    logger.error(f"NLLB: Invalid language codes {src_lang}->{src_code} or {tgt_lang}->{tgt_code}")
                    return "ERROR: Invalid language code for NLLB model"
                # NLLB requires setting src_lang and forcing target BOS token
                # Check if tokenizer has src_lang attribute before setting
                if hasattr(self.tokenizer, 'src_lang'):
                    self.tokenizer.src_lang = src_code
                else:
                    logger.warning(f"Tokenizer for {self.hf_id} does not have src_lang attribute. Proceeding without setting it.")

                # Check if lang_code_to_id exists before using it
                # Also check for the deprecation warning context if possible
                forced_bos_token_id = None
                if hasattr(self.tokenizer, 'lang_code_to_id') and tgt_code in self.tokenizer.lang_code_to_id:
                     forced_bos_token_id = self.tokenizer.lang_code_to_id[tgt_code]
                     logger.debug(f"NLLB: Using src_code={src_code}, tgt_code={tgt_code}, forced_bos_id={forced_bos_token_id}")
                     generate_args = {"forced_bos_token_id": forced_bos_token_id}
                # Handle potential future removal or alternative ways if lang_code_to_id is gone
                elif hasattr(self.tokenizer, 'convert_tokens_to_ids'):
                     try:
                          # Attempt to get the ID using a standard method if lang_code_to_id is missing/deprecated
                          bos_token = f">>{tgt_code}<<" # Common format, adjust if NLLB uses different special token format
                          forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(bos_token)
                          # Check if it's a valid ID (not UNK token id)
                          if forced_bos_token_id != self.tokenizer.unk_token_id:
                              logger.debug(f"NLLB: Using src_code={src_code}, tgt_code={tgt_code}, forced_bos_id={forced_bos_token_id} (via convert_tokens_to_ids)")
                              generate_args = {"forced_bos_token_id": forced_bos_token_id}
                          else:
                              logger.warning(f"Could not get valid BOS token ID for '{tgt_code}' using convert_tokens_to_ids. Proceeding without forced_bos_token_id.")
                              generate_args = {}
                     except Exception as conv_err:
                          logger.warning(f"Error getting BOS token ID for '{tgt_code}': {conv_err}. Proceeding without forced_bos_token_id.")
                          generate_args = {}
                else:
                     logger.warning(f"Could not find target language code '{tgt_code}' in tokenizer attributes for {self.hf_id}. Proceeding without forced_bos_token_id.")
                     generate_args = {}

            else: # Assume MarianMT or similar (doesn't need special codes)
                logger.debug(f"MarianMT: No special language codes needed for generation.")
                generate_args = {}
            # -----------------------------

            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()} # Move tensors to device

            # Generate translation
            translated_tokens = self.model.generate(
                **inputs,
                max_length=512, # Adjust max_length if needed
                num_beams=4, # Example: Add beam search for potentially better quality
                early_stopping=True,
                **generate_args # Add model-specific args (e.g., forced_bos_token_id for NLLB)
            )

            # Decode the generated tokens
            translated_text = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            end_time = time.time()
            logger.info(f"{self.hf_id} translation successful in {end_time - start_time:.2f}s. Result: '{translated_text[:100]}...'")
            return translated_text

        except Exception as e:
            logger.error(f"Error during {self.hf_id} translation from {src_lang} to {tgt_lang}: {e}", exc_info=True)
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                 try:
                      from torch.mps import empty_cache
                      empty_cache()
                 except ImportError:
                      logger.warning("Could not import torch.mps.empty_cache(). MPS memory might not be fully cleared.")
            gc.collect() # Trigger garbage collection
            logger.info(f"Model {self.hf_id} unloaded.")


class Translator:
    """
    Translation interface for EchoLang.
    Selects appropriate model (MarianMT first, then NLLB fallback) based on language pair.
    Manages loading/unloading models to conserve memory.
    """

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        # Store loaded models keyed by their hf_id
        self._loaded_models: Dict[str, TransformersModel] = {}
        self._current_model_id: Optional[str] = None
        # --- REMOVED ERRONEOUS LINE: self.translation_model = TransformersModel(...) ---

    def _get_model_config(self, src_lang: str, tgt_lang: str) -> Optional[Dict]:
        """Determine the best model config for the given language pair."""
        pair_key = f"{src_lang}-{tgt_lang}"

        # Prioritize MarianMT
        if pair_key in config.MARIAN_CONFIG:
            logger.info(f"Using MarianMT model for pair: {pair_key}")
            return config.MARIAN_CONFIG[pair_key]

        # Fallback to NLLB
        nllb_config_key = config.DEFAULT_NLLB_MODEL_KEY
        nllb_config = config.NLLB_CONFIG.get(nllb_config_key)
        if nllb_config:
             src_nllb = LanguageCode.code_to_nllb_code(src_lang)
             tgt_nllb = LanguageCode.code_to_nllb_code(tgt_lang)
             if src_nllb and tgt_nllb:
                  logger.info(f"Using NLLB fallback model ({nllb_config_key}) for pair: {pair_key}")
                  return nllb_config
             else:
                  logger.warning(f"Language pair {pair_key} not directly supported by NLLB codes. Cannot use NLLB fallback.")
                  return None

        logger.warning(f"No translation model configuration found for pair: {pair_key}")
        return None

    def _load_or_get_model(self, model_config: Dict) -> Optional[TransformersModel]:
        """Loads the specified model if not already loaded, unloads others if needed."""
        hf_id = model_config.get("hf_id")
        if not hf_id:
            logger.error("Cannot load model: Config missing 'hf_id'.")
            return None

        # Unload previous model if different
        if self._current_model_id and self._current_model_id != hf_id:
            if self._current_model_id in self._loaded_models:
                logger.info(f"Switching translation models: Unloading {self._current_model_id}")
                self._loaded_models[self._current_model_id].unload_model()
                # Remove from dict to ensure complete reload next time? Maybe safer.
                del self._loaded_models[self._current_model_id]
            self._current_model_id = None

        # Load new model if needed
        if hf_id not in self._loaded_models:
            logger.info(f"Attempting to load translation model: {hf_id}")
            try:
                # Pass the specific model_config (contains hf_id and model_type)
                model_wrapper = TransformersModel(model_config, self.model_manager)
                model_wrapper.load_model()
                self._loaded_models[hf_id] = model_wrapper
                self._current_model_id = hf_id
                logger.info(f"Successfully loaded and cached model wrapper for {hf_id}")
            except Exception as e:
                logger.error(f"Failed to load model {hf_id}: {e}", exc_info=True)
                if self._current_model_id == hf_id: # Ensure it's not marked as current
                    self._current_model_id = None
                # Remove failed entry if it was added
                if hf_id in self._loaded_models:
                    del self._loaded_models[hf_id]
                return None # Failed to load
        else:
             # Model already loaded, ensure it's set as current
             self._current_model_id = hf_id
             logger.debug(f"Using already loaded model: {hf_id}")


        return self._loaded_models.get(hf_id)


    def translate(self,
                  text: str,
                  src_lang: str = LanguageCode.ENGLISH,
                  tgt_lang: str = LanguageCode.HINDI) -> Dict:
        """
        Translate text from source language to target language.
        Selects the appropriate model (MarianMT or NLLB fallback).

        Args:
            text: Text to translate
            src_lang: Source language code (internal)
            tgt_lang: Target language code (internal)

        Returns:
            Dictionary with translation results
        """
        translated_text = f"ERROR: Could not translate from {src_lang} to {tgt_lang}" # Default error
        error_message = None

        if src_lang == tgt_lang:
            logger.warning(f"Source ({src_lang}) and Target ({tgt_lang}) languages are the same. Skipping translation.")
            translated_text = text # Return original text
        else:
            model_config = self._get_model_config(src_lang, tgt_lang)
            if not model_config:
                error_message = f"No suitable translation model found for {src_lang} -> {tgt_lang}."
                logger.error(error_message)
                translated_text = f"ERROR: {error_message}"
            else:
                model_wrapper = self._load_or_get_model(model_config)
                if not model_wrapper:
                    error_message = f"Failed to load model for {src_lang} -> {tgt_lang} ({model_config.get('hf_id')})."
                    logger.error(error_message)
                    translated_text = f"ERROR: {error_message}"
                else:
                    # Perform translation using the loaded model wrapper
                    translated_text = model_wrapper.translate(text, src_lang, tgt_lang)
                    if translated_text.startswith("ERROR:"):
                        error_message = translated_text # Capture error from lower level

        # Check if translation returned an error message
        has_error = translated_text.startswith("ERROR:")

        return {
            "original_text": text,
            "translated_text": translated_text if not has_error else "",
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "error": error_message or (translated_text if has_error else None) # Report specific error
        }