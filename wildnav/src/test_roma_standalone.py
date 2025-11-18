"""
Standalone RoMa test script
Tests RoMa installation and basic matching functionality
This should be run in Colab with GPU for speed
"""
import torch
import numpy as np

print("="*70)
print("RoMa Standalone Test")
print("="*70)

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Step 1: Import RoMa
print("\n[Step 1] Importing RoMa...")
try:
    from romatch import roma_outdoor
    print("✓ RoMa imported successfully")
except ImportError as e:
    print(f"✗ Failed to import RoMa: {e}")
    print("\nInstall RoMa with: pip install romatch")
    exit(1)

# Step 2: Load RoMa model
print("\n[Step 2] Loading RoMa model...")
try:
    roma_model = roma_outdoor(device=device)
    print("✓ RoMa model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load RoMa model: {e}")
    exit(1)

# Step 3: Load test images
print("\n[Step 3] Loading test images...")
query_path = '../assets/map/1_query_image.png'
sat_path = '../assets/map/sat_map_00.png'

import os
if not os.path.exists(query_path):
    print(f"✗ Query image not found: {query_path}")
    exit(1)
if not os.path.exists(sat_path):
    print(f"✗ Satellite image not found: {sat_path}")
    exit(1)

print(f"  Query: {query_path}")
print(f"  Satellite: {sat_path}")
print("✓ Images found")

# Step 4: Perform matching
print("\n[Step 4] Matching images with RoMa...")
try:
    import time
    start_time = time.time()

    # RoMa's match method returns dense warp and certainty
    warp, certainty = roma_model.match(query_path, sat_path, device=device)

    elapsed = time.time() - start_time
    print(f"✓ Matching completed in {elapsed:.2f}s")
    print(f"  Warp shape: {warp.shape}")
    print(f"  Certainty shape: {certainty.shape}")

except Exception as e:
    print(f"✗ Matching failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 5: Sample sparse matches from dense prediction
print("\n[Step 5] Sampling sparse correspondences...")
try:
    # Sample matches from the dense warp field
    # This converts dense prediction to sparse keypoint matches
    matches, certainty_sampled = roma_model.sample(warp, certainty)

    print(f"✓ Sampled correspondences")
    print(f"  Matches shape: {matches.shape}")
    print(f"  Certainty shape: {certainty_sampled.shape}")

    # Get image dimensions for coordinate conversion
    from PIL import Image
    query_img = Image.open(query_path)
    sat_img = Image.open(sat_path)
    H_query, W_query = query_img.size[1], query_img.size[0]
    H_sat, W_sat = sat_img.size[1], sat_img.size[0]

    # Convert to pixel coordinates
    kpts_query, kpts_sat = roma_model.to_pixel_coordinates(
        matches, H_query, W_query, H_sat, W_sat
    )

    print(f"✓ Converted to pixel coordinates")
    print(f"  Query keypoints shape: {kpts_query.shape}")
    print(f"  Satellite keypoints shape: {kpts_sat.shape}")

    # Count valid matches (with sufficient certainty)
    certainty_threshold = 0.5
    valid_matches = certainty_sampled > certainty_threshold
    num_valid = valid_matches.sum().item()

    print(f"\n  Valid matches (certainty > {certainty_threshold}): {num_valid}")
    print(f"  Mean certainty: {certainty_sampled.mean().item():.4f}")
    print(f"  Max certainty: {certainty_sampled.max().item():.4f}")

    # Show sample match coordinates
    if num_valid > 0:
        print(f"\n  Sample matches (first 3):")
        for i in range(min(3, num_valid)):
            idx = torch.where(valid_matches)[0][i].item()
            print(f"    Match {i+1}: Query({kpts_query[idx, 0]:.1f}, {kpts_query[idx, 1]:.1f}) -> "
                  f"Sat({kpts_sat[idx, 0]:.1f}, {kpts_sat[idx, 1]:.1f}), "
                  f"certainty={certainty_sampled[idx].item():.4f}")

except Exception as e:
    print(f"✗ Sampling failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ RoMa installation: OK")
print(f"✓ Model loading: OK")
print(f"✓ Image matching: OK ({elapsed:.2f}s)")
print(f"✓ Sparse sampling: OK ({num_valid} valid matches)")
print("\nRoMa is ready to use!")
