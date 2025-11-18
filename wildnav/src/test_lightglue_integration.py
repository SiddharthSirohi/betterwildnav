"""
Test LightGlue integration with DINOv2 features
"""
import torch
from lightglue import LightGlue
from lightglue.utils import load_image
from dinov2_extractor import DINOv2Extractor

print("=" * 60)
print("Testing LightGlue + DINOv2 Integration")
print("=" * 60)

device = 'cpu'

# Initialize DINOv2 extractor
print(f"\n1. Initializing DINOv2 extractor...")
extractor = DINOv2Extractor(
    model_name='dinov2_vits14',
    max_keypoints=2048,
    device=device
)
print("   ✓ DINOv2 initialized")

# Initialize LightGlue matcher
print(f"\n2. Initializing LightGlue matcher with 'disk' features (128-dim default)...")
matcher = LightGlue(features='disk').eval().to(device)
print(f"   ✓ LightGlue initialized")
print(f"   - LightGlue input_dim: {matcher.conf.input_dim}")

# Load two test images
print(f"\n3. Loading test images...")
image0 = load_image('../assets/map/1_query_image.png').to(device)
image1 = load_image('../assets/map/sat_map_00.png').to(device)
print(f"   ✓ Images loaded")

# Extract features
print(f"\n4. Extracting features from both images...")
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)
print(f"   ✓ Features extracted")
print(f"   - Image 0 descriptors shape: {feats0['descriptors'].shape}")
print(f"   - Image 1 descriptors shape: {feats1['descriptors'].shape}")

# Check dimension compatibility
desc_dim = feats0['descriptors'].shape[-1]
lightglue_dim = matcher.conf.input_dim
print(f"\n5. Dimension check:")
print(f"   - Descriptor dimension: {desc_dim}")
print(f"   - LightGlue expects: {lightglue_dim}")

if desc_dim == lightglue_dim:
    print("   ✓ Dimensions match!")
else:
    print(f"   ✗ ERROR: Dimension mismatch!")
    print(f"   Need to configure LightGlue with input_dim={desc_dim}")

# Try matching
print(f"\n6. Attempting to match features...")
try:
    matches01 = matcher({'image0': feats0, 'image1': feats1})
    print(f"   ✓ Matching successful!")
    print(f"   - Number of matches: {matches01['matches'].shape[0]}")
except AssertionError as e:
    print(f"   ✗ Matching failed with AssertionError")
    print(f"   Error: {e}")
except Exception as e:
    print(f"   ✗ Matching failed with error: {type(e).__name__}")
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
