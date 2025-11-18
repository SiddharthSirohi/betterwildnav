"""
DINOv2 Feature Extractor Wrapper for LightGlue Compatibility
Extracts dense features from DINOv2 (via PyTorch Hub) and converts them to keypoint format for LightGlue
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import ssl
import urllib.request

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context


class DINOv2Extractor:
    """
    Wrapper for DINOv2 (PyTorch Hub) that produces LightGlue-compatible features.
    Extracts dense features and samples keypoints on a grid.
    """

    def __init__(self, model_name='dinov2_vits14', max_keypoints=2048, device='cpu'):
        """
        Initialize DINOv2 feature extractor.

        Args:
            model_name: PyTorch Hub model name. Options:
                        - 'dinov2_vits14' (21M params, default)
                        - 'dinov2_vitb14' (86M params)
                        - 'dinov2_vitl14' (300M params)
                        - 'dinov2_vitg14' (1.1B params)
                        Add '_reg' suffix for versions with registers (e.g., 'dinov2_vits14_reg')
            max_keypoints: Maximum number of keypoints to extract
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.max_keypoints = max_keypoints
        self.model_name = model_name

        print(f"Loading DINOv2 model: {model_name} from PyTorch Hub")

        # Load model from PyTorch Hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(device)
        self.model.eval()

        # Get patch size from model config (DINOv2 uses 14x14 patches for all models)
        self.patch_size = 14

        print(f"DINOv2 model loaded successfully (patch_size={self.patch_size})")

    def extract(self, image_tensor):
        """
        Extract features from an image in LightGlue-compatible format.

        Args:
            image_tensor: PyTorch tensor (C, H, W) or (1, C, H, W) normalized to [0, 1]

        Returns:
            dict with 'keypoints', 'descriptors', 'scores', 'image_size'
        """
        with torch.no_grad():
            # Ensure batch dimension exists
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Get original dimensions (C, H, W format expected)
            _, _, h, w = image_tensor.shape

            # DINOv2 requires dimensions to be multiples of patch_size (14)
            # Round down to nearest multiple
            h_new = (h // self.patch_size) * self.patch_size
            w_new = (w // self.patch_size) * self.patch_size

            # Resize if needed
            if h != h_new or w != w_new:
                image_tensor = F.interpolate(
                    image_tensor,
                    size=(h_new, w_new),
                    mode='bilinear',
                    align_corners=False
                )

            # DINOv2 expects RGB images normalized with ImageNet stats
            # LightGlue's load_image already normalizes to [0, 1], but DINOv2 needs specific normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

            # Apply ImageNet normalization
            image_normalized = (image_tensor - mean) / std

            # Extract features using DINOv2
            # forward_features returns patch tokens (without CLS token)
            features = self.model.forward_features(image_normalized)

            # Get patch embeddings - DINOv2 returns dict with 'x_norm_patchtokens'
            if isinstance(features, dict):
                patch_tokens = features['x_norm_patchtokens']  # (batch, num_patches, hidden_dim)
            else:
                # Some models return just the tensor
                patch_tokens = features[:, 1:]  # Remove CLS token if present

            # Calculate grid dimensions based on patch size
            # Use the adjusted dimensions (h_new, w_new) that are multiples of patch_size
            _, _, h_model, w_model = image_normalized.shape
            num_patches = patch_tokens.shape[1]
            grid_h = h_model // self.patch_size
            grid_w = w_model // self.patch_size

            # Ensure we have the right number of patches
            assert num_patches == grid_h * grid_w, f"Patch mismatch: {num_patches} != {grid_h * grid_w}"

            # Reshape to grid: (batch, hidden_dim, grid_h, grid_w)
            hidden_dim = patch_tokens.shape[2]
            features_grid = patch_tokens.permute(0, 2, 1).reshape(1, hidden_dim, grid_h, grid_w)

            # Upsample features to higher resolution for finer keypoint localization
            # Quarter resolution provides good balance between detail and efficiency
            features_upsampled = F.interpolate(
                features_grid,
                size=(h // 4, w // 4),
                mode='bilinear',
                align_corners=False
            )

            # Extract keypoints and descriptors (use original h, w for scaling back)
            keypoints, descriptors, scores = self._extract_keypoints_and_descriptors(
                features_upsampled, h, w
            )

            # Project descriptors to 128 dimensions (to match LightGlue 'disk' default)
            # DINOv2 ViT-S produces 384-dim descriptors
            target_dim = 128
            if descriptors.shape[1] != target_dim:
                # Simple linear projection using a random orthogonal matrix (deterministic for consistency)
                torch.manual_seed(42)  # For reproducibility
                if not hasattr(self, 'projection'):
                    self.projection = torch.nn.Linear(descriptors.shape[1], target_dim, bias=False).to(self.device)
                    # Normalize the projection matrix
                    with torch.no_grad():
                        self.projection.weight.data = F.normalize(self.projection.weight.data, dim=1)
                    self.projection.eval()

                descriptors = self.projection(descriptors)
                # Normalize descriptors
                descriptors = F.normalize(descriptors, dim=1)

            # Add batch dimension to match LightGlue's expected format
            return {
                'keypoints': keypoints.unsqueeze(0),  # (1, N, 2)
                'descriptors': descriptors.unsqueeze(0),  # (1, N, 128)
                'scores': scores.unsqueeze(0),  # (1, N)
                'image_size': torch.tensor([[w, h]], dtype=torch.float32, device=self.device)  # (1, 2)
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
