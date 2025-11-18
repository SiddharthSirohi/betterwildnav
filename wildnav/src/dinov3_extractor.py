"""
DINOv3 Feature Extractor Wrapper for LightGlue Compatibility
Extracts dense features from DINOv3 and converts them to keypoint format for LightGlue
"""
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image


class DINOv3Extractor:
    """
    Wrapper for DINOv3 that produces LightGlue-compatible features.
    Extracts dense features and samples keypoints on a grid.
    """

    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", max_keypoints=2048, device='cpu'):
        """
        Initialize DINOv3 feature extractor.

        Args:
            model_name: Hugging Face model identifier
            max_keypoints: Maximum number of keypoints to extract
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.max_keypoints = max_keypoints
        self.model_name = model_name

        print(f"Loading DINOv3 model: {model_name}")

        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        # Get patch size from config (usually 16 for ViT/16 models)
        self.patch_size = self.model.config.patch_size if hasattr(self.model.config, 'patch_size') else 16

        print(f"DINOv3 model loaded successfully (patch_size={self.patch_size})")

    def extract(self, image_tensor):
        """
        Extract features from an image in LightGlue-compatible format.

        Args:
            image_tensor: PyTorch tensor (C, H, W) normalized to [0, 1]

        Returns:
            dict with 'keypoints', 'descriptors', 'scores', 'image_size'
        """
        with torch.no_grad():
            # Convert tensor to PIL Image for processor
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Denormalize from [0, 1] to [0, 255] for PIL
            image_np = (image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)

            # Get original dimensions
            h, w = pil_image.size[1], pil_image.size[0]

            # Process image
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

            # Extract features from DINOv3
            outputs = self.model(**inputs)

            # Get patch embeddings (last_hidden_state)
            # Shape: (batch, num_patches, hidden_dim)
            features = outputs.last_hidden_state

            # Remove CLS token (first token)
            features = features[:, 1:, :]  # (1, num_patches, hidden_dim)

            # Calculate grid dimensions
            num_patches = features.shape[1]
            grid_size = int(np.sqrt(num_patches))

            # Reshape to grid: (1, hidden_dim, grid_h, grid_w)
            features = features.permute(0, 2, 1).reshape(1, -1, grid_size, grid_size)

            # Upsample features to image resolution for finer details
            # This helps with more accurate keypoint localization
            features_upsampled = F.interpolate(
                features,
                size=(h // 4, w // 4),  # Quarter resolution for efficiency
                mode='bilinear',
                align_corners=False
            )

            # Extract keypoints and descriptors
            keypoints, descriptors, scores = self._extract_keypoints_and_descriptors(
                features_upsampled, h, w
            )

            return {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores,
                'image_size': torch.tensor([w, h], dtype=torch.float32, device=self.device)
            }

    def _extract_keypoints_and_descriptors(self, features, orig_h, orig_w):
        """
        Extract keypoints and descriptors from dense feature map.
        Uses grid sampling for uniform coverage.

        Args:
            features: Dense feature map (1, C, H, W)
            orig_h, orig_w: Original image dimensions

        Returns:
            keypoints: (N, 2) tensor of x,y coordinates
            descriptors: (N, C) tensor of feature descriptors
            scores: (N,) tensor of confidence scores
        """
        batch, channels, feat_h, feat_w = features.shape

        # Calculate grid spacing based on desired number of keypoints
        num_points_per_dim = int(np.sqrt(self.max_keypoints))

        # Create uniform grid of keypoints
        y_coords = torch.linspace(0, feat_h - 1, num_points_per_dim, device=self.device)
        x_coords = torch.linspace(0, feat_w - 1, num_points_per_dim, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Flatten grid
        grid_y = grid_y.reshape(-1)
        grid_x = grid_x.reshape(-1)

        # Extract features at grid locations using bilinear interpolation
        # Normalize coordinates to [-1, 1] for grid_sample
        norm_x = 2.0 * grid_x / (feat_w - 1) - 1.0
        norm_y = 2.0 * grid_y / (feat_h - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)

        # Sample features
        sampled_features = F.grid_sample(
            features,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # (1, C, 1, N)

        descriptors = sampled_features.squeeze(0).squeeze(1).t()  # (N, C)

        # Scale keypoint coordinates to original image size
        scale_x = orig_w / feat_w
        scale_y = orig_h / feat_h

        keypoints_x = grid_x * scale_x
        keypoints_y = grid_y * scale_y
        keypoints = torch.stack([keypoints_x, keypoints_y], dim=-1)  # (N, 2)

        # Calculate scores based on descriptor norm (simple heuristic)
        scores = torch.norm(descriptors, dim=1)
        scores = scores / scores.max()  # Normalize to [0, 1]

        # Limit to max_keypoints if needed
        if keypoints.shape[0] > self.max_keypoints:
            # Keep top scoring keypoints
            top_indices = torch.argsort(scores, descending=True)[:self.max_keypoints]
            keypoints = keypoints[top_indices]
            descriptors = descriptors[top_indices]
            scores = scores[top_indices]

        return keypoints, descriptors, scores

    def eval(self):
        """Set model to evaluation mode (for compatibility with LightGlue API)"""
        self.model.eval()
        return self

    def to(self, device):
        """Move model to device (for compatibility with LightGlue API)"""
        self.device = device
        self.model = self.model.to(device)
        return self
