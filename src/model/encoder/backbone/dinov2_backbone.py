"""
DINOv2 Backbone for depth prediction.
Uses pretrained DINOv2 to extract image features for monocular depth estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DINOv2Backbone(nn.Module):
    """
    DINOv2 feature extractor.

    Args:
        model_name: DINOv2 model name, options:
            - 'dinov2_vits14': ViT-Small, 384-dim features
            - 'dinov2_vitb14': ViT-Base, 768-dim features
            - 'dinov2_vitl14': ViT-Large, 1024-dim features
        freeze_backbone: whether to freeze DINOv2 parameters (recommended True)
        feature_dim: output feature dimension (corresponds to model_name)
    """

    def __init__(
        self,
        model_name: str = 'dinov2_vits14',
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        # DINOv2 feature dimension mapping
        self.feature_dims = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
        }

        if model_name not in self.feature_dims:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Choose from {list(self.feature_dims.keys())}"
            )

        self.feature_dim = self.feature_dims[model_name]

        # Load pretrained DINOv2 model
        print(f"Loading DINOv2 model: {model_name}")
        self.dinov2 = torch.hub.load(
            'facebookresearch/dinov2',
            model_name,
            pretrained=True
        )

        # Freeze backbone parameters
        if freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()
            print(f"DINOv2 backbone frozen (requires_grad=False)")
        else:
            print(f"DINOv2 backbone trainable")

        # DINOv2 patch size (default 14)
        self.patch_size = 14

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: input image (B, 3, H, W)

        Returns:
            features: feature map (B, feature_dim, H', W')
                where H' = H // patch_size, W' = W // patch_size
        """
        B, C, H, W = x.shape

        # Ensure 3-channel input
        assert C == 3, f"Expected 3 channels, got {C}"

        # DINOv2 requires input size to be a multiple of patch_size
        # Pad if necessary
        H_pad = ((H + self.patch_size - 1) // self.patch_size) * self.patch_size
        W_pad = ((W + self.patch_size - 1) // self.patch_size) * self.patch_size

        if H_pad != H or W_pad != W:
            # Use reflection padding to maintain edge continuity
            pad_h = H_pad - H
            pad_w = W_pad - W
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # DINOv2 forward pass
        if self.freeze_backbone:
            with torch.no_grad():
                # get_intermediate_layers returns patch tokens (excluding CLS token)
                features = self.dinov2.forward_features(x)
                # features['x_norm_patchtokens']: (B, num_patches, feature_dim)
                patch_features = features['x_norm_patchtokens']
        else:
            features = self.dinov2.forward_features(x)
            patch_features = features['x_norm_patchtokens']

        # Compute patch grid size (based on padded size)
        H_patches = H_pad // self.patch_size
        W_patches = W_pad // self.patch_size

        # Reshape: (B, num_patches, C) -> (B, C, H', W')
        try:
            features_2d = rearrange(
                patch_features,
                'b (h w) c -> b c h w',
                h=H_patches,
                w=W_patches
            )
        except Exception as e:
            print(f"Error reshaping features: {e}")
            print(f"Input shape (after padding): {x.shape}")
            print(f"Patch features shape: {patch_features.shape}")
            print(f"Trying to reshape to: ({B}, {self.feature_dim}, {H_patches}, {W_patches})")
            raise

        # Crop back to original feature size if padding was applied
        H_feat_original = (H + self.patch_size - 1) // self.patch_size
        W_feat_original = (W + self.patch_size - 1) // self.patch_size

        if H_patches != H_feat_original or W_patches != W_feat_original:
            features_2d = features_2d[:, :, :H_feat_original, :W_feat_original]

        return features_2d

    def get_feature_dim(self) -> int:
        """Return feature dimension."""
        return self.feature_dim

    def get_output_size(self, input_height: int, input_width: int) -> tuple:
        """
        Compute output feature map size for a given input size.

        Args:
            input_height: input image height
            input_width: input image width

        Returns:
            (output_height, output_width): output feature map size
        """
        H_out = (input_height + self.patch_size - 1) // self.patch_size
        W_out = (input_width + self.patch_size - 1) // self.patch_size
        return H_out, W_out


if __name__ == "__main__":
    # Test code
    print("Testing DINOv2Backbone...")

    for model_name in ['dinov2_vits14']:  # Test small model only
        print(f"\n{'='*50}")
        print(f"Testing {model_name}")
        print(f"{'='*50}")

        backbone = DINOv2Backbone(
            model_name=model_name,
            freeze_backbone=True
        )

        batch_size = 2
        H, W = 256, 320
        x = torch.randn(batch_size, 3, H, W)

        print(f"Input shape: {x.shape}")

        with torch.no_grad():
            features = backbone(x)

        print(f"Output shape: {features.shape}")
        print(f"Feature dim: {backbone.get_feature_dim()}")

        expected_H, expected_W = backbone.get_output_size(H, W)
        print(f"Expected output size: ({expected_H}, {expected_W})")

        assert features.shape[0] == batch_size
        assert features.shape[1] == backbone.get_feature_dim()
        print("Test passed!")

    print("\n" + "="*50)
    print("All tests passed successfully!")
