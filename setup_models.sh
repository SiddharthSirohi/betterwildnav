#!/bin/bash
# Setup script for downloading OmniGlue models
# For use in Google Colab or any Linux environment

set -e  # Exit on error

echo "Setting up OmniGlue models for WildNav integration..."

# Create models directory
mkdir -p models
cd models

echo "===================================="
echo "1. Downloading SuperPoint weights..."
echo "===================================="
if [ ! -d "sp_v6" ]; then
    git clone https://github.com/rpautrat/SuperPoint.git
    mv SuperPoint/pretrained_models/sp_v6.tgz .
    rm -rf SuperPoint
    tar zxvf sp_v6.tgz
    rm sp_v6.tgz
    echo "✓ SuperPoint downloaded successfully"
else
    echo "✓ SuperPoint already exists"
fi

echo ""
echo "===================================="
echo "2. Downloading DINOv2 weights..."
echo "===================================="
if [ ! -f "dinov2_vitb14_pretrain.pth" ]; then
    wget --show-progress https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth
    echo "✓ DINOv2 downloaded successfully (~330MB)"
else
    echo "✓ DINOv2 already exists"
fi

echo ""
echo "===================================="
echo "3. Downloading OmniGlue weights..."
echo "===================================="
if [ ! -d "og_export" ]; then
    wget --show-progress https://storage.googleapis.com/omniglue/og_export.zip
    unzip -q og_export.zip
    rm og_export.zip
    echo "✓ OmniGlue downloaded successfully"
else
    echo "✓ OmniGlue already exists"
fi

cd ..

echo ""
echo "===================================="
echo "Model setup complete!"
echo "===================================="
echo "Directory structure:"
echo "models/"
echo "  ├── sp_v6/              (SuperPoint)"
echo "  ├── dinov2_vitb14_pretrain.pth  (DINOv2)"
echo "  └── og_export/          (OmniGlue)"
echo ""
echo "Total size: ~375MB"
