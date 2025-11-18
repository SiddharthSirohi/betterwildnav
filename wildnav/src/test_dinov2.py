"""
Simple test script to verify DINOv2 feature extraction works correctly
"""
import torch
from lightglue.utils import load_image
from dinov2_extractor import DINOv2Extractor

print("=" * 60)
print("Testing DINOv2 Feature Extraction")
print("=" * 60)

# Initialize DINOv2 extractor
device = 'cpu'
print(f"\n1. Initializing DINOv2 extractor on {device}...")
extractor = DINOv2Extractor(
    model_name='dinov2_vits14',
    max_keypoints=2048,
    device=device
)
print("   ✓ DINOv2 extractor initialized successfully")

# Load test image
test_image_path = '../assets/map/1_query_image.png'
print(f"\n2. Loading test image: {test_image_path}")
image_tensor = load_image(test_image_path).to(device)
print(f"   ✓ Image loaded with shape: {image_tensor.shape}")

# Extract features
print(f"\n3. Extracting features...")
features = extractor.extract(image_tensor)
print("   ✓ Features extracted successfully")

# Print feature shapes
print(f"\n4. Feature details:")
print(f"   - Keypoints shape: {features['keypoints'].shape}")
print(f"   - Descriptors shape: {features['descriptors'].shape}")
print(f"   - Scores shape: {features['scores'].shape}")
print(f"   - Image size: {features['image_size']}")

# Verify dimensions
batch_size, num_keypoints, descriptor_dim = features['descriptors'].shape
print(f"\n5. Verification:")
print(f"   - Batch size: {batch_size}")
print(f"   - Number of keypoints: {num_keypoints}")
print(f"   - Descriptor dimension: {descriptor_dim}")

if descriptor_dim == 256:
    print("   ✓ Descriptor dimension is correct (256)")
else:
    print(f"   ✗ ERROR: Descriptor dimension should be 256, got {descriptor_dim}")

if batch_size == 1:
    print("   ✓ Batch dimension is correct (1)")
else:
    print(f"   ✗ ERROR: Batch size should be 1, got {batch_size}")

print("\n" + "=" * 60)
print("DINOv2 Test Complete!")
print("=" * 60)
