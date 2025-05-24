import os
import sys
import hashlib
import requests
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, List, Union

# Model configurations
MODELS_CONFIG = {
    'camus': {
        'filename': 'CAMUS_diffusion_model.pt', # Standard nnU-Net models might be .model or .pth
        'url': 'https://github.com/devilog4n/Echo_Weights/raw/main/CAMUS_diffusion_model.pt', # Direct raw link
        'sha256': None,  # Add SHA256 after first download
        'save_dir': 'models'
    },
    'best': {
        'filename': 'checkpoint_best.pth',
        'url': 'https://github.com/devilog4n/Echo_Weights/releases/download/v1.0.0/checkpoint_best.pth',
        'sha256': None,  # Add SHA256 after first download
        'save_dir': 'models'
    },
    'final': {
        'filename': 'checkpoint_final.pth',
        'url': 'https://github.com/devilog4n/Echo_Weights/releases/download/v1.0.0/checkpoint_final.pth',
        'sha256': None,  # Add SHA256 after first download
        'save_dir': 'models'
    }
}

def calculate_sha256(file_path: str, chunk_size: int = 8192) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    except Exception as e:
        print(f"Error calculating checksum for {file_path}: {str(e)}")
        return ""

def download_file(url: str, destination: str, expected_sha256: Optional[str] = None) -> bool:
    """
    Download a file with progress bar and optional checksum verification.

    Args:
        url: URL of the file to download
        destination: Local path to save the file
        expected_sha256: Optional SHA256 checksum to verify the downloaded file

    Returns:
        bool: True if download and verification were successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        # Check if file already exists
        if os.path.exists(destination):
            if expected_sha256:
                actual_sha256 = calculate_sha256(destination)
                if actual_sha256 == expected_sha256:
                    print(f"✓ File exists and checksum matches: {os.path.basename(destination)}")
                    return True
                print(f"Existing file checksum doesn't match. Redownloading {os.path.basename(destination)}...")
            else:
                print(f"✓ File exists (no checksum verification): {os.path.basename(destination)}")
                return True

        print(f"\nDownloading {os.path.basename(destination)}...")
        print(f"URL: {url}")

        # Make the request with streaming
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get total file size from headers
        total_size = int(response.headers.get('content-length', 0))

        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ascii=True,
            ncols=80
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                bar.update(size)

        # Verify checksum if provided
        if expected_sha256:
            actual_sha256 = calculate_sha256(destination)
            if actual_sha256 != expected_sha256:
                print(f"✗ Checksum verification failed! Expected {expected_sha256}, got {actual_sha256}")
                os.remove(destination)
                return False
            print("✓ Checksum verified successfully!")

        print(f"✓ Successfully downloaded to: {os.path.abspath(destination)}")
        return True

    except Exception as e:
        print(f"✗ Error downloading {os.path.basename(destination)}: {str(e)}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def download_model(model_name: str = 'all') -> Dict[str, str]:
    """
    Download one or all models.

    Args:
        model_name: Name of the model to download or 'all' for all models

    Returns:
        Dict containing the paths of downloaded models
    """
    downloaded_paths = {}

    # Determine which models to download
    if model_name.lower() == 'all':
        models_to_download = MODELS_CONFIG.items()
    else:
        if model_name not in MODELS_CONFIG:
            print(f"Error: Unknown model '{model_name}'. Available models: {', '.join(MODELS_CONFIG.keys())}")
            return {}
        models_to_download = [(model_name, MODELS_CONFIG[model_name])]

    print(f"\n{'='*50}")
    print(f"Downloading {len(models_to_download)} model{'s' if len(models_to_download) > 1 else ''}")
    print("="*50)

    # Download each model
    for name, config in models_to_download:
        save_path = os.path.join(config['save_dir'], config['filename'])
        success = download_file(
            url=config['url'],
            destination=save_path,
            expected_sha256=config['sha256']
        )

        if success:
            downloaded_paths[name] = save_path

            # Calculate and print SHA256 for the first time if not set
            if not config['sha256']:
                sha256 = calculate_sha256(save_path)
                print(f"\nFirst-time download. Add this SHA256 checksum for {name} model:")
                print(f"'sha256': '{sha256}',")

    return downloaded_paths

def list_available_models() -> None:
    """List all available models and their download status."""
    print("\nAvailable models:")
    print("-" * 50)
    for name, config in MODELS_CONFIG.items():
        save_path = os.path.join(config['save_dir'], config['filename'])
        exists = os.path.exists(save_path)
        status = "✓ Downloaded" if exists else "✗ Not downloaded"
        print(f"{name:10} {status}")
        print(f"  Path: {os.path.abspath(save_path)}")
        if exists:
            print(f"  Size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
            if config['sha256']:
                print(f"  SHA256: {config['sha256']}")
        print()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download Echo model weights')
    parser.add_argument('model', nargs='?', default='all',
                       help='Model name to download (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available models and exit')

    args = parser.parse_args()

    if args.list:
        list_available_models()
    else:
        download_model(args.model)
