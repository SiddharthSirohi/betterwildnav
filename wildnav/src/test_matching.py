"""
Comprehensive matching test to diagnose why we're getting 0 matches
"""
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image
from dinov2_extractor import DINOv2Extractor
from dinov2_hybrid_extractor import DINOv2HybridExtractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load test images - query and one satellite map
query_path = '../assets/map/1_query_image.png'
satellite_path = '../assets/map/sat_map_00.png'

print("Loading images...")
query_tensor = load_image(query_path).to(device)
sat_tensor = load_image(satellite_path).to(device)
print(f"Query shape: {query_tensor.shape}")
print(f"Satellite shape: {sat_tensor.shape}\n")

# Test 1: Pure SuperPoint (baseline - should work)
print("="*70)
print("TEST 1: Pure SuperPoint (Baseline)")
print("="*70)
sp_extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
sp_matcher = LightGlue(features='superpoint').eval().to(device)

sp_feats0 = sp_extractor.extract(query_tensor)
sp_feats1 = sp_extractor.extract(sat_tensor)

print(f"Query keypoints: {sp_feats0['keypoints'].shape}")
print(f"Satellite keypoints: {sp_feats1['keypoints'].shape}")

sp_matches = sp_matcher({'image0': sp_feats0, 'image1': sp_feats1})
print(f"Matches output type: {type(sp_matches)}")
print(f"Matches keys: {sp_matches.keys()}")
# Get valid matches
if 'matches0' in sp_matches:
    valid_mask = sp_matches['matches0'][0] > -1
    sp_num_matches = valid_mask.sum().item()
else:
    sp_num_matches = 0
print(f"✓ SuperPoint matches found: {sp_num_matches}\n")

# Test 2: Hybrid SuperPoint+DINOv2
print("="*70)
print("TEST 2: Hybrid SuperPoint+DINOv2")
print("="*70)
hybrid_extractor = DINOv2HybridExtractor(
    model_name='dinov2_vits14',
    max_keypoints=2048,
    device=device
).eval()
hybrid_matcher = LightGlue(features='superpoint').eval().to(device)

hybrid_feats0 = hybrid_extractor.extract(query_tensor)
hybrid_feats1 = hybrid_extractor.extract(sat_tensor)

print(f"Query keypoints: {hybrid_feats0['keypoints'].shape}")
print(f"Satellite keypoints: {hybrid_feats1['keypoints'].shape}")

# Check feature compatibility
print("\nFeature structure comparison:")
print(f"  SuperPoint keys: {sp_feats0.keys()}")
print(f"  Hybrid keys: {hybrid_feats0.keys()}")

print("\nDescriptor comparison:")
print(f"  SuperPoint desc shape: {sp_feats0['descriptors'].shape}")
print(f"  Hybrid desc shape: {hybrid_feats0['descriptors'].shape}")
print(f"  SuperPoint desc norm (first 3): {torch.norm(sp_feats0['descriptors'][0, :3], dim=1)}")
print(f"  Hybrid desc norm (first 3): {torch.norm(hybrid_feats0['descriptors'][0, :3], dim=1)}")

# Check if descriptors are different
desc_diff = torch.norm(sp_feats0['descriptors'] - hybrid_feats0['descriptors'])
print(f"  Descriptor difference L2 norm: {desc_diff:.4f}")

try:
    hybrid_matches = hybrid_matcher({'image0': hybrid_feats0, 'image1': hybrid_feats1})
    if 'matches0' in hybrid_matches:
        valid_mask = hybrid_matches['matches0'][0] > -1
        hybrid_num_matches = valid_mask.sum().item()
    else:
        hybrid_num_matches = 0
    print(f"\n✓ Hybrid matches found: {hybrid_num_matches}")
except Exception as e:
    print(f"\n✗ Hybrid matching failed with error: {e}")
    hybrid_num_matches = None

# Test 3: Check LightGlue's internal processing
print("\n" + "="*70)
print("TEST 3: Detailed LightGlue Processing")
print("="*70)

# Manually check what LightGlue expects
print("\nLightGlue 'superpoint' config:")
print(f"  Input dim: {sp_matcher.conf.input_dim}")
print(f"  Descriptor dim: {sp_matcher.conf.descriptor_dim}")

print("\nChecking if hybrid features pass LightGlue's validation...")
# Try to manually prepare features as LightGlue expects
data = {'image0': hybrid_feats0, 'image1': hybrid_feats1}

# Check tensor devices
print(f"  Query descriptors device: {hybrid_feats0['descriptors'].device}")
print(f"  Sat descriptors device: {hybrid_feats1['descriptors'].device}")
print(f"  Matcher device: {next(hybrid_matcher.parameters()).device}")

# Test 4: Pure DINOv2 grid-based (for comparison)
print("\n" + "="*70)
print("TEST 4: Pure DINOv2 Grid-based")
print("="*70)
dino_extractor = DINOv2Extractor(
    model_name='dinov2_vits14',
    max_keypoints=2048,
    device=device
)
dino_matcher = LightGlue(features='disk').eval().to(device)

dino_feats0 = dino_extractor.extract(query_tensor)
dino_feats1 = dino_extractor.extract(sat_tensor)

print(f"Query keypoints: {dino_feats0['keypoints'].shape}")
print(f"Satellite keypoints: {dino_feats1['keypoints'].shape}")

try:
    dino_matches = dino_matcher({'image0': dino_feats0, 'image1': dino_feats1})
    if 'matches0' in dino_matches:
        valid_mask = dino_matches['matches0'][0] > -1
        dino_num_matches = valid_mask.sum().item()
    else:
        dino_num_matches = 0
    print(f"✓ DINOv2 grid matches found: {dino_num_matches}")
except Exception as e:
    print(f"✗ DINOv2 matching failed with error: {e}")
    dino_num_matches = None

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"SuperPoint (baseline):     {sp_num_matches} matches")
print(f"Hybrid SuperPoint+DINOv2:  {hybrid_num_matches if 'hybrid_num_matches' in locals() else 'FAILED'} matches")
print(f"Pure DINOv2 grid:          {dino_num_matches if 'dino_num_matches' in locals() else 'FAILED'} matches")
