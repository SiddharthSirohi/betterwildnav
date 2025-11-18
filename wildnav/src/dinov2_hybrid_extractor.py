"""
Hybrid DINOv2 + SuperPoint Feature Extractor
Uses SuperPoint for keypoint detection and DINOv2 for descriptor extraction
"""
import torch
import torch.nn.functional as F
import numpy as np
import ssl

# Fix SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context


class DINOv2HybridExtractor:
    """
    Hybrid extractor that uses:
    - SuperPoint for keypoint detection (corners, edges - good for matching)
    - DINOv2 for descriptor extraction (semantic features - more robust)
    """

    def __init__(self, model_name='dinov2_vits14', max_keypoints=2048, device='cpu'):
        """
        Initialize hybrid extractor.

        Args:
            model_name: DINOv2 model name from PyTorch Hub
            max_keypoints: Maximum number of keypoints to extract
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.max_keypoints = max_keypoints
        self.model_name = model_name

        # Initialize SuperPoint for keypoint detection
        print(f"Loading SuperPoint for keypoint detection...")
        from lightglue import SuperPoint
        self.keypoint_detector = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)

        # Initialize DINOv2 for descriptor extraction
        print(f"Loading DINOv2 model: {model_name} for descriptor extraction...")
        self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino_model = self.dino_model.to(device)
        self.dino_model.eval()

        self.patch_size = 14  # DINOv2 patch size

        print(f"Hybrid extractor ready: SuperPoint (keypoints) + DINOv2 (descriptors)")

    def extract(self, image_tensor):
        """
        Extract features using hybrid approach.

        Args:
            image_tensor: PyTorch tensor (C, H, W) or (1, C, H, W) normalized to [0, 1]

        Returns:
            dict with 'keypoints', 'descriptors', 'keypoints_scores', 'image_size'
        """
        with torch.no_grad():
            # Ensure batch dimension
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            _, _, h, w = image_tensor.shape

            # Step 1: Detect keypoints using SuperPoint
            sp_features = self.keypoint_detector.extract(image_tensor)
            keypoints = sp_features['keypoints'][0]  # (N, 2) - remove batch dim
            sp_scores = sp_features['keypoints_scores'][0] if 'keypoints_scores' in sp_features else None

            num_kpts = keypoints.shape[0]
            print(f"  SuperPoint detected {num_kpts} keypoints")

            # Step 2: Extract DINOv2 descriptors at those keypoint locations
            # Prepare image for DINOv2
            h_new = (h // self.patch_size) * self.patch_size
            w_new = (w // self.patch_size) * self.patch_size

            if h != h_new or w != w_new:
                image_resized = F.interpolate(
                    image_tensor,
                    size=(h_new, w_new),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                image_resized = image_tensor

            # Apply ImageNet normalization for DINOv2
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            image_normalized = (image_resized - mean) / std

            # Extract DINOv2 features
            features = self.dino_model.forward_features(image_normalized)

            if isinstance(features, dict):
                patch_tokens = features['x_norm_patchtokens']
            else:
                patch_tokens = features[:, 1:]

            # Reshape to spatial grid
            _, _, h_model, w_model = image_normalized.shape
            grid_h = h_model // self.patch_size
            grid_w = w_model // self.patch_size
            hidden_dim = patch_tokens.shape[2]

            features_grid = patch_tokens.permute(0, 2, 1).reshape(1, hidden_dim, grid_h, grid_w)

            # Upsample to full resolution for better sampling
            features_upsampled = F.interpolate(
                features_grid,
                size=(h, w),
                mode='bilinear',
                align_corners=False
            )

            # Sample DINOv2 descriptors at SuperPoint keypoint locations
            # Normalize keypoint coordinates to [-1, 1] for grid_sample
            norm_x = 2.0 * keypoints[:, 0] / (w - 1) - 1.0
            norm_y = 2.0 * keypoints[:, 1] / (h - 1) - 1.0
            grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)

            # Sample features at keypoint locations
            sampled_descriptors = F.grid_sample(
                features_upsampled,
                grid,
                mode='bilinear',
                padding_mode='border',
                align_corners=True
            )  # (1, C, 1, N)

            descriptors = sampled_descriptors.squeeze(2).squeeze(0).t()  # (N, C)

            # Project to 256-dim (SuperPoint compatibility)
            target_dim = 256
            if descriptors.shape[1] != target_dim:
                if not hasattr(self, 'projection'):
                    torch.manual_seed(42)
                    self.projection = torch.nn.Linear(descriptors.shape[1], target_dim, bias=False).to(self.device)
                    with torch.no_grad():
                        self.projection.weight.data = F.normalize(self.projection.weight.data, dim=1)
                    self.projection.eval()

                descriptors = self.projection(descriptors)

            # L2 normalize descriptors (standard for feature matching)
            descriptors = F.normalize(descriptors, dim=1)

            # Return in LightGlue format
            return {
                'keypoints': keypoints.unsqueeze(0),  # (1, N, 2)
                'descriptors': descriptors.unsqueeze(0),  # (1, N, 256)
                'keypoints_scores': sp_scores.unsqueeze(0) if sp_scores is not None else torch.ones(1, num_kpts, device=self.device),
                'image_size': torch.tensor([[w, h]], dtype=torch.float32, device=self.device)
            }

    def eval(self):
        """Set to evaluation mode"""
        self.keypoint_detector.eval()
        self.dino_model.eval()
        return self

    def to(self, device):
        """Move to device"""
        self.device = device
        self.keypoint_detector = self.keypoint_detector.to(device)
        self.dino_model = self.dino_model.to(device)
        return self
