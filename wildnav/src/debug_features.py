"""
Debug script to compare SuperPoint vs DINOv2 feature extraction
"""
import torch
from lightglue import SuperPoint
from lightglue.utils import load_image
from dinov2_extractor import DINOv2Extractor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Load test image
image_path = '../assets/map/1_query_image.png'
image_tensor = load_image(image_path).to(device)
print(f"Image shape: {image_tensor.shape}\n")

# Extract with SuperPoint
print("="*60)
print("SuperPoint Extraction")
print("="*60)
sp_extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
sp_features = sp_extractor.extract(image_tensor)

print(f"Keypoints shape: {sp_features['keypoints'].shape}")
print(f"Keypoints sample (first 3): \n{sp_features['keypoints'][0, :3]}")
print(f"Descriptors shape: {sp_features['descriptors'].shape}")
print(f"Descriptors dtype: {sp_features['descriptors'].dtype}")
print(f"Descriptor norm (first): {torch.norm(sp_features['descriptors'][0, 0]):.4f}")
print(f"Descriptor stats: min={sp_features['descriptors'].min():.4f}, max={sp_features['descriptors'].max():.4f}, mean={sp_features['descriptors'].mean():.4f}")
print(f"Scores shape: {sp_features['keypoints_scores'].shape if 'keypoints_scores' in sp_features else 'N/A'}")
if 'image_size' in sp_features:
    print(f"Image size: {sp_features['image_size']}")
print(f"\nAll keys: {sp_features.keys()}")

# Extract with DINOv2
print("\n" + "="*60)
print("DINOv2 Extraction")
print("="*60)
dino_extractor = DINOv2Extractor(model_name='dinov2_vits14', max_keypoints=2048, device=device)
dino_features = dino_extractor.extract(image_tensor)

print(f"Keypoints shape: {dino_features['keypoints'].shape}")
print(f"Keypoints sample (first 3): \n{dino_features['keypoints'][0, :3]}")
print(f"Descriptors shape: {dino_features['descriptors'].shape}")
print(f"Descriptors dtype: {dino_features['descriptors'].dtype}")
print(f"Descriptor norm (first): {torch.norm(dino_features['descriptors'][0, 0]):.4f}")
print(f"Descriptor stats: min={dino_features['descriptors'].min():.4f}, max={dino_features['descriptors'].max():.4f}, mean={dino_features['descriptors'].mean():.4f}")
print(f"Scores shape: {dino_features['scores'].shape}")
print(f"Image size: {dino_features['image_size']}")
print(f"\nAll keys: {dino_features.keys()}")

# Check descriptor norms
print("\n" + "="*60)
print("Descriptor Analysis")
print("="*60)
sp_desc_norms = torch.norm(sp_features['descriptors'][0], dim=1)
dino_desc_norms = torch.norm(dino_features['descriptors'][0], dim=1)

print(f"SuperPoint descriptor norms: min={sp_desc_norms.min():.4f}, max={sp_desc_norms.max():.4f}, mean={sp_desc_norms.mean():.4f}")
print(f"DINOv2 descriptor norms: min={dino_desc_norms.min():.4f}, max={dino_desc_norms.max():.4f}, mean={dino_desc_norms.mean():.4f}")

# Check if descriptors are unit-normalized
print(f"\nSuperPoint descriptors unit-normalized? {torch.allclose(sp_desc_norms, torch.ones_like(sp_desc_norms), atol=0.01)}")
print(f"DINOv2 descriptors unit-normalized? {torch.allclose(dino_desc_norms, torch.ones_like(dino_desc_norms), atol=0.01)}")
