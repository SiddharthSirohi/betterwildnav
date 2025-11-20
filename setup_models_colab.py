#!/usr/bin/env python3
"""
Setup script for downloading OmniGlue models in Google Colab
Saves models to Google Drive for persistence across sessions
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description=""):
    """Run shell command and handle errors"""
    if description:
        print(f"\n{'='*50}")
        print(f"{description}")
        print('='*50)

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    if result.stdout:
        print(result.stdout)
    return True

def setup_models():
    """Download and setup all required models"""

    # Check if running in Colab
    try:
        from google.colab import drive
        IN_COLAB = True
        print("✓ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("✓ Running locally")

    # Setup paths
    if IN_COLAB:
        # Mount Google Drive
        print("\nMounting Google Drive...")
        drive.mount('/content/drive', force_remount=False)

        # Use persistent location in Drive
        models_base = Path('/content/drive/MyDrive/wildnav_models')
        models_base.mkdir(exist_ok=True)
        print(f"✓ Models will be saved to: {models_base}")
        print("  (Persistent across Colab sessions)")

        # Create symlink in working directory
        local_models = Path('models')
        if local_models.exists() and local_models.is_symlink():
            local_models.unlink()
        if not local_models.exists():
            os.symlink(models_base, local_models)
            print(f"✓ Created symlink: ./models -> {models_base}")
    else:
        models_base = Path('models')
        models_base.mkdir(exist_ok=True)
        print(f"✓ Models will be saved to: {models_base}")

    os.chdir(models_base)

    # 1. Download SuperPoint
    sp_path = models_base / 'sp_v6'
    if sp_path.exists():
        print("\n✓ SuperPoint already exists, skipping download")
    else:
        print("\n" + "="*50)
        print("1. Downloading SuperPoint weights (~5MB)")
        print("="*50)
        run_command("git clone https://github.com/rpautrat/SuperPoint.git")
        run_command("mv SuperPoint/pretrained_models/sp_v6.tgz .")
        run_command("rm -rf SuperPoint")
        run_command("tar zxvf sp_v6.tgz")
        run_command("rm sp_v6.tgz")
        if sp_path.exists():
            print("✓ SuperPoint downloaded successfully")
        else:
            print("✗ SuperPoint download failed")
            return False

    # 2. Download DINOv2
    dino_path = models_base / 'dinov2_vitb14_pretrain.pth'
    if dino_path.exists():
        print("\n✓ DINOv2 already exists, skipping download")
    else:
        print("\n" + "="*50)
        print("2. Downloading DINOv2 weights (~330MB)")
        print("="*50)
        print("This may take 2-5 minutes...")
        success = run_command(
            "wget -q --show-progress https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
        )
        if success and dino_path.exists():
            print("✓ DINOv2 downloaded successfully")
        else:
            print("✗ DINOv2 download failed")
            return False

    # 3. Download OmniGlue
    og_path = models_base / 'og_export'
    if og_path.exists():
        print("\n✓ OmniGlue already exists, skipping download")
    else:
        print("\n" + "="*50)
        print("3. Downloading OmniGlue weights (~40MB)")
        print("="*50)
        success = run_command(
            "wget -q --show-progress https://storage.googleapis.com/omniglue/og_export.zip"
        )
        if success:
            run_command("unzip -q og_export.zip")
            run_command("rm og_export.zip")
            if og_path.exists():
                print("✓ OmniGlue downloaded successfully")
            else:
                print("✗ OmniGlue download failed")
                return False

    # Summary
    print("\n" + "="*50)
    print("✓ Model setup complete!")
    print("="*50)
    print(f"\nModels location: {models_base}")
    print("\nDirectory structure:")
    print("models/")
    print("  ├── sp_v6/                      (SuperPoint)")
    print("  ├── dinov2_vitb14_pretrain.pth  (DINOv2)")
    print("  └── og_export/                  (OmniGlue)")
    print("\nTotal size: ~375MB")

    if IN_COLAB:
        print("\n📌 These models are saved in your Google Drive")
        print("   and will persist across Colab sessions!")

    return True

if __name__ == "__main__":
    success = setup_models()
    sys.exit(0 if success else 1)
