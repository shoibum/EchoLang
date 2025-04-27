# src/utils/model_utils.py
import os
import torch
import hashlib
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

try:
    from .. import config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    import config

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Handles model downloading (for non-HF models like XTTS),
    caching, device selection, and paths.
    """

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ModelManager.

        Args:
            models_dir: Path to the directory for storing *local* models (like XTTS).
                        Defaults to MODELS_DIR from config.py.
        """
        self.models_dir = Path(models_dir) if models_dir is not None else config.MODELS_DIR
        try:
            self.models_dir.mkdir(exist_ok=True, parents=True)
        except OSError as e:
            logger.error(f"Failed to create models directory {self.models_dir}: {e}", exc_info=True)
            raise
        # Use device/dtype from central config
        self.device = torch.device(config.APP_DEVICE)
        self.torch_dtype = config.APP_TORCH_DTYPE
        logger.info(f"ModelManager initialized. Using models directory: {self.models_dir}")
        logger.info(f"Using device: {self.device} with dtype: {self.torch_dtype}")

    # _get_device is removed as logic is moved to config.py

    def download_file(self, url: str, output_path: Path,
                      expected_sha256: Optional[str] = None) -> Path:
        """
        Download a file with progress bar and optional checksum verification.
        (Still used for XTTS model files).
        """
        output_path.parent.mkdir(exist_ok=True, parents=True)
        logger.debug(f"Requesting download from {url} to {output_path}")

        # Check checksum if file exists and hash is provided
        if output_path.exists() and expected_sha256:
            logger.debug(f"File {output_path} exists. Verifying checksum...")
            sha256 = hashlib.sha256()
            try:
                with open(output_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256.update(byte_block)
                calculated_hash = sha256.hexdigest()
                if calculated_hash == expected_sha256:
                    logger.info(f"Checksum matches for existing file: {output_path}. Skipping download.")
                    return output_path
                else:
                    logger.warning(f"Checksum mismatch for {output_path}. Expected {expected_sha256}, got {calculated_hash}. Redownloading.")
            except Exception as e:
                 logger.warning(f"Could not verify checksum for {output_path}: {e}. Redownloading.", exc_info=True)

        # Download with progress bar
        try:
            response = requests.get(url, stream=True, timeout=60) # Increased timeout
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024 * 4 # 4 KB

            logger.info(f"Downloading {url} to {output_path} ({total_size / (1024*1024):.2f} MB)")
            with open(output_path, "wb") as file, tqdm(
                total=total_size, unit="B", unit_scale=True, desc=output_path.name, leave=False
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    file.write(data)
                    progress_bar.update(len(data))

            if total_size != 0 and progress_bar.n != total_size:
                 logger.warning(f"Download finished, but size mismatch for {output_path}. Expected {total_size}, got {progress_bar.n}.")

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed for {url}: {e}", exc_info=True)
            if output_path.exists(): output_path.unlink(missing_ok=True)
            raise
        except Exception as e:
            logger.error(f"An error occurred during download: {e}", exc_info=True)
            if output_path.exists(): output_path.unlink(missing_ok=True)
            raise

        # Verify checksum if provided after download
        if expected_sha256:
            logger.debug(f"Verifying checksum post-download for {output_path}...")
            sha256 = hashlib.sha256()
            try:
                with open(output_path, "rb") as f:
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256.update(byte_block)
                calculated_hash = sha256.hexdigest()
                if calculated_hash != expected_sha256:
                    logger.error(f"Checksum verification failed after download! Expected: {expected_sha256}, Got: {calculated_hash}")
                    output_path.unlink(missing_ok=True) # Delete corrupted download
                    raise ValueError(f"Checksum mismatch for {output_path}.")
                else:
                    logger.info(f"Checksum verified successfully for {output_path}.")
            except Exception as e:
                 logger.error(f"Could not verify checksum after download for {output_path}: {e}", exc_info=True)
                 raise

        return output_path

    def get_model_path(self, model_relative_path: Union[str, Path]) -> Path:
        """Get absolute path for a model relative to the *local* models directory."""
        return self.models_dir / model_relative_path

    def ensure_model_downloaded(self, model_relative_path: Union[str, Path], url: str,
                              expected_sha256: Optional[str] = None) -> Path:
        """Ensure *local* model (e.g., XTTS) is downloaded, downloading if necessary."""
        output_path = self.get_model_path(model_relative_path)
        # Always attempt download/checksum verification if hash is provided or file missing
        if not output_path.exists() or expected_sha256:
            logger.info(f"Checking/downloading local model file {output_path} from {url}.")
            try:
                return self.download_file(url, output_path, expected_sha256)
            except Exception as e:
                logger.error(f"Failed to ensure model file {output_path} is downloaded: {e}", exc_info=True)
                raise
        else:
            logger.debug(f"Local model file {output_path} already exists and no checksum provided. Skipping download.")
            return output_path