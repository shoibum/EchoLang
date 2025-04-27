# src/utils/model_utils.py
import os
import torch
import hashlib
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any, Union

class ModelManager:
    """Handles model downloading, caching, and loading."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.device = self._get_device()
        
    def _get_device(self) -> torch.device:
        """Determine the best available device, preferring MPS on Apple Silicon."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
            
    def download_file(self, url: str, output_path: Path, 
                      expected_sha256: Optional[str] = None) -> Path:
        """
        Download a file with progress bar and optional checksum verification.
        
        Args:
            url: URL to download
            output_path: Where to save the file
            expected_sha256: Expected SHA256 hash (optional)
            
        Returns:
            Path to downloaded file
        """
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Skip if file exists and checksum matches
        if output_path.exists() and expected_sha256:
            sha256 = hashlib.sha256()
            with open(output_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256.update(byte_block)
            if sha256.hexdigest() == expected_sha256:
                print(f"File already exists and checksum matches: {output_path}")
                return output_path
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 KB
        
        print(f"Downloading {url} to {output_path}")
        with open(output_path, "wb") as file, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=output_path.name
        ) as progress_bar:
            for data in response.iter_content(block_size):
                file.write(data)
                progress_bar.update(len(data))
                
        # Verify checksum if provided
        if expected_sha256:
            sha256 = hashlib.sha256()
            with open(output_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256.update(byte_block)
            calculated_hash = sha256.hexdigest()
            if calculated_hash != expected_sha256:
                raise ValueError(
                    f"Checksum verification failed. Expected: {expected_sha256}, Got: {calculated_hash}"
                )
                
        return output_path
        
    def get_model_path(self, model_name: str) -> Path:
        """Get path for a model in the models directory."""
        return self.models_dir / model_name
        
    def ensure_model_downloaded(self, model_name: str, url: str, 
                              expected_sha256: Optional[str] = None) -> Path:
        """Ensure model is downloaded, downloading if necessary."""
        output_path = self.get_model_path(model_name)
        if not output_path.exists():
            return self.download_file(url, output_path, expected_sha256)
        return output_path