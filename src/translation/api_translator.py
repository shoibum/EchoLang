# src/translation/api_translator.py
import logging
from typing import Optional, Dict
from pathlib import Path
import requests # Make sure 'requests' is installed (it's in your requirements.txt)
import json

# Use relative imports for config
try:
    from .. import config
    from ..utils.language import LanguageCode # For language codes if needed
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config
    from utils.language import LanguageCode


logger = logging.getLogger(__name__)

class LibreTranslateAPITranslator:
    """
    Handles translation using a LibreTranslate API endpoint via HTTP requests.
    """
    def __init__(self):
        self.api_url = config.LIBRETRANSLATE_URL
        if not self.api_url.endswith('/'):
            self.api_url += '/'
        self.translate_endpoint = self.api_url + "translate"
        logger.info(f"LibreTranslateAPITranslator initialized. API endpoint: {self.translate_endpoint}")
        # Check connectivity? Optional.
        # try:
        #     response = requests.get(self.api_url + "languages", timeout=5)
        #     response.raise_for_status()
        #     logger.info("Successfully connected to LibreTranslate API languages endpoint.")
        # except requests.exceptions.RequestException as e:
        #     logger.warning(f"Could not connect to LibreTranslate API ({self.api_url}): {e}")

    def translate(self,
                  text: str,
                  src_lang: str = LanguageCode.ENGLISH,
                  tgt_lang: str = LanguageCode.HINDI) -> Dict:
        """
        Translate text using the LibreTranslate API.

        Args:
            text: Text to translate
            src_lang: Source language code (e.g., 'en', 'hi', 'kn')
            tgt_lang: Target language code (e.g., 'en', 'hi', 'kn')

        Returns:
            Dictionary with translation results, consistent with pipeline expectations.
        """
        translated_text = ""
        error_message = None

        if not text or not text.strip():
            error_message = "Input text is empty."
            logger.warning(error_message)
        elif src_lang == tgt_lang:
            logger.warning(f"Source ({src_lang}) and Target ({tgt_lang}) languages are the same. Skipping API call.")
            translated_text = text # Return original text
        else:
            logger.info(f"Requesting LibreTranslate translation: {src_lang} -> {tgt_lang} for text: '{text[:50]}...'")
            headers = {'Content-Type': 'application/json'}
            payload = {
                'q': text,
                'source': src_lang,
                'target': tgt_lang,
                'format': 'text'
                # 'api_key': 'YOUR_API_KEY' # Add if using an instance that requires a key
            }

            try:
                response = requests.post(
                    self.translate_endpoint,
                    headers=headers,
                    data=json.dumps(payload), # Send data as JSON string
                    timeout=30 # Add a timeout (e.g., 30 seconds)
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                result_data = response.json()

                if isinstance(result_data, dict) and 'translatedText' in result_data:
                    translated_text = result_data['translatedText']
                    # Check for detected language if source was 'auto' (not used here currently)
                    # detected_lang_info = result_data.get('detectedLanguage')
                    # if detected_lang_info:
                    #    logger.info(f"LibreTranslate detected source: {detected_lang_info.get('language')} (Conf: {detected_lang_info.get('confidence')})")
                    logger.info(f"LibreTranslate translation successful. Result: '{translated_text[:100]}...'")
                else:
                    error_message = "Invalid response format from LibreTranslate API."
                    logger.error(f"{error_message} Response: {result_data}")

            except requests.exceptions.Timeout:
                 error_message = "API call timed out."
                 logger.error(error_message)
            except requests.exceptions.RequestException as e:
                error_message = f"API call failed: {e}"
                logger.error(error_message, exc_info=True)
                # Check status code for rate limiting (often 429)
                if e.response is not None and e.response.status_code == 429:
                     error_message = "LibreTranslate API rate limit likely exceeded."
                     logger.warning(error_message)
            except json.JSONDecodeError:
                error_message = "Failed to decode API response (not valid JSON)."
                logger.error(f"{error_message} Response text: {response.text[:200]}...")
            except Exception as e:
                error_message = f"An unexpected error occurred during API translation: {type(e).__name__} - {e}"
                logger.error(error_message, exc_info=True)


        # Format output consistently with the pipeline
        has_error = error_message is not None
        return {
            "original_text": text,
            "translated_text": translated_text if not has_error else "",
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "error": error_message
        }