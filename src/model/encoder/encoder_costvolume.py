from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from collections import OrderedDict

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import (
    BackboneMultiview,
)
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg

from ...global_cfg import get_cfg

from .epipolar.epipolar_sampler import EpipolarSampler
from ..encodings.positional_encoding import PositionalEncoding


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool

class SurfelAdaption(nn.Module):
    """Computes adaptive attention window size for each pixel."""
    def __init__(self, s=1.0, min_window=3, max_window=5, eps=1e-6):
        super().__init__()
        self.s = s
        self.min_window = min_window
        self.max_window = max_window
        self.eps = eps
        # Allowed window sizes: 3x3, 4x4, 5x5
        self.allowed_windows = [3, 4, 5]

    def compute_spatial_frequency(self, depths, fx, fy):
        """
        Compute spatial frequency: v = fx * fy / d^2
        depths: [B, V, H*W, srf, 1]
        fx, fy: [B, V]
        returns: [B, V, H*W]
        """
        # Take depth of the first surface
        d = depths[:, :, :, 0, 0]  # [B, V, H*W]

        # Expand fx, fy dimensions to match
        fx = fx.unsqueeze(-1)  # [B, V, 1]
        fy = fy.unsqueeze(-1)  # [B, V, 1]

        # Compute spatial frequency, add eps to avoid division by zero
        v = (fx * fy) / (d ** 2 + self.eps)  # [B, V, H*W]

        return v

    def compute_window_size(self, spatial_freq):
        """
        Compute window size: window_size = floor(sqrt(1 + s^2 / v^2))
        spatial_freq: [B, V, H*W]
        returns: [B*V, H, W] integer tensor
        """
        # Compute raw window size
        raw_window = torch.sqrt(1 + (self.s ** 2) / (spatial_freq ** 2 + self.eps))
        raw_window = torch.floor(raw_window)

        # Clamp to allowed window size range
        window_sizes = torch.zeros_like(raw_window)
        for allowed_ws in self.allowed_windows:
            mask = raw_window >= allowed_ws
            window_sizes[mask] = allowed_ws

        # Ensure at least minimum window size
        window_sizes = torch.clamp(window_sizes, min=self.min_window, max=self.max_window)

        return window_sizes.long()

    def forward(self, depths, intrinsics, h, w):
        """
        depths: [B, V, H*W, srf, 1]
        intrinsics: [B, V, 3, 3]
        h, w: image height and width
        returns: [B*V, H, W] window sizes
        """
        b, v = depths.shape[:2]

        # Extract focal lengths
        fx = intrinsics[:, :, 0, 0]  # [B, V]
        fy = intrinsics[:, :, 1, 1]  # [B, V]

        # Compute spatial frequency
        spatial_freq = self.compute_spatial_frequency(depths, fx, fy)  # [B, V, H*W]

        # Compute window sizes
        window_sizes = self.compute_window_size(spatial_freq)  # [B, V, H*W]

        # Reshape to image space
        window_sizes = rearrange(window_sizes, "b v (h w) -> (b v) h w", h=h, w=w)

        return window_sizes


class FeatureAggregation(nn.Module):
    """Feature refinement using dynamic windows."""
    def __init__(self, feature_dim, normal_dim, output_dim, hidden_dim=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else feature_dim
        self.normal_dim = normal_dim

        # Projection layers for local self-attention (for normal features)
        self.normal_query_proj = nn.Conv2d(feature_dim, self.hidden_dim, kernel_size=1)
        self.normal_key_proj = nn.Conv2d(feature_dim, self.hidden_dim, kernel_size=1)
        self.normal_value_proj = nn.Conv2d(feature_dim, self.hidden_dim, kernel_size=1)

        # Local self-attention output projection
        self.normal_output_proj = nn.Conv2d(self.hidden_dim, feature_dim, kernel_size=1)

        # Scale factor
        self.scale = self.hidden_dim ** -0.5

        # Normalization layers
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)

        # Local Cross Attention layers
        self.img_proj = nn.Conv2d(feature_dim, self.hidden_dim, kernel_size=1)
        self.refined_normal_k_proj = nn.Conv2d(feature_dim, self.hidden_dim, kernel_size=1)
        self.refined_normal_v_proj = nn.Conv2d(feature_dim, self.hidden_dim, kernel_size=1)

        # Cross Attention output projection
        self.cross_output_proj = nn.Conv2d(self.hidden_dim, feature_dim, kernel_size=1)

        # Final output layer
        self.final_proj = nn.Conv2d(feature_dim, output_dim, kernel_size=1)

        # FFN layer
        self.ffn = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(feature_dim * 4, feature_dim, kernel_size=1)
        )

        self.norm3 = nn.LayerNorm(feature_dim)

        # Upsampler
        self.upsampler = nn.Sequential(
            nn.Conv2d(normal_dim, feature_dim-6, 3, 1, 1),
            nn.Upsample(
                scale_factor=4,
                mode="bilinear",
                align_corners=True,
            ),
            nn.GELU(),
        )
    def _local_attention_for_normal_dynamic(self, x, window_sizes):
        """Apply dynamic window local self-attention to normal features (memory-efficient version)."""
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        batch_size, _, height, width = x.shape

        # Compute queries, keys and values
        q = self.normal_query_proj(x_norm)
        k = self.normal_key_proj(x_norm)
        v = self.normal_value_proj(x_norm)

        # Use maximum window size for a single unfold (5x5)
        max_ws = 5
        padding = max_ws // 2
        k_unfolded = F.unfold(k, kernel_size=max_ws, padding=padding)
        v_unfolded = F.unfold(v, kernel_size=max_ws, padding=padding)

        # Reshape: [B, H_dim, max_ws^2, H, W]
        k_unfolded = k_unfolded.reshape(batch_size, self.hidden_dim, max_ws**2, height, width)
        v_unfolded = v_unfolded.reshape(batch_size, self.hidden_dim, max_ws**2, height, width)

        # Create dynamic mask: [B, 1, max_ws^2, H, W]
        attention_mask = self._create_window_mask(window_sizes, max_ws, height, width, batch_size, x.device)

        q_expanded = q.unsqueeze(2)  # [B, H_dim, 1, H, W]

        # Compute attention scores
        attn_scores = (q_expanded * k_unfolded).sum(dim=1, keepdim=True) * self.scale  # [B, 1, max_ws^2, H, W]

        # Apply mask (set invalid positions to -inf)
        attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=2)  # [B, 1, max_ws^2, H, W]

        # Weighted sum
        output = (attn_weights * v_unfolded).sum(dim=2)  # [B, H_dim, H, W]

        # Output projection
        output = self.normal_output_proj(output)

        # Residual connection
        output = output + x

        return output

    def _create_window_mask(self, window_sizes, max_ws, height, width, batch_size, device):
        """
        Create dynamic window mask.
        window_sizes: [B, H, W] window size per pixel
        max_ws: maximum window size
        returns: [B, 1, max_ws^2, H, W] bool mask
        """
        # Create relative position grid
        half_max = max_ws // 2
        y_offset = torch.arange(-half_max, half_max + 1, device=device)
        x_offset = torch.arange(-half_max, half_max + 1, device=device)
        yy, xx = torch.meshgrid(y_offset, x_offset, indexing='ij')

        # Compute Chebyshev distance from each neighbor to center
        distances = torch.maximum(torch.abs(yy), torch.abs(xx))  # [max_ws, max_ws]
        distances = distances.reshape(-1)  # [max_ws^2]

        # Expand dimensions to match
        distances = distances.view(1, 1, max_ws**2, 1, 1)  # [1, 1, max_ws^2, 1, 1]
        window_sizes = window_sizes.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H, W]

        # Mask: positions with distance <= half_window are True
        half_window = (window_sizes - 1) // 2
        mask = distances <= half_window  # [B, 1, max_ws^2, H, W]

        return mask
        
    def _local_cross_attention_dynamic(self, img_feats, refined_normal_feats, window_sizes):
        """Localized cross attention with dynamic window sizes - memory optimized."""
        img_feats_norm = self.norm2(img_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        normal_feats_norm = self.norm2(refined_normal_feats.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        batch_size, feat_dim, height, width = img_feats.shape

        # Project features
        img_proj = self.img_proj(img_feats_norm)
        normal_k = self.refined_normal_k_proj(normal_feats_norm)
        normal_v = self.refined_normal_v_proj(normal_feats_norm)

        # Single unfold using max window size (5x5)
        max_ws = 5
        padding = max_ws // 2
        normal_k_unfolded = F.unfold(normal_k, kernel_size=max_ws, padding=padding)
        normal_v_unfolded = F.unfold(normal_v, kernel_size=max_ws, padding=padding)

        # Reshape
        normal_k_unfolded = normal_k_unfolded.reshape(batch_size, self.hidden_dim, max_ws**2, height, width)
        normal_v_unfolded = normal_v_unfolded.reshape(batch_size, self.hidden_dim, max_ws**2, height, width)

        # Create dynamic mask
        attention_mask = self._create_window_mask(window_sizes, max_ws, height, width, batch_size, img_feats.device)

        img_q = img_proj.unsqueeze(2)  # [B*V, H_dim, 1, H, W]

        # Compute attention scores
        attn_scores = (img_q * normal_k_unfolded).sum(dim=1, keepdim=True) * self.scale

        # Apply mask
        attn_scores = attn_scores.masked_fill(~attention_mask, float('-inf'))

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=2)

        # Weighted sum
        output = (attn_weights * normal_v_unfolded).sum(dim=2)

        # Output projection
        output = self.cross_output_proj(output)

        # Residual connection
        output = output + img_feats

        return output
    
    def _ffn_block(self, x):
        x_norm = self.norm3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        ffn_output = self.ffn(x_norm)
        output = ffn_output + x
        return output

    def forward(self, raw_in_feat, normal_features, context, window_sizes, estimated_normals):
        """
        Process image features and normal features using dynamic windows.

        Args:
            raw_in_feat: raw image features [B*V, C, H, W]
            normal_features: normal transformer features [B*V, C, H, W]
            context: context information
            window_sizes: dynamic window sizes [B*V, H, W]
            estimated_normals: estimated surface normals [B*V, 3, H, W]

        Returns:
            refined_features: refined features [B*V, output_dim, H, W]
        """
        # 1. Prepare normal features
        images = rearrange(context["image"], "b v c h w -> (b v) c h w")
        normal_features = self.upsampler(normal_features)
        normal_features = torch.cat((normal_features, images, estimated_normals), dim=1)

        # 2. Apply dynamic window local self-attention to normal features
        refined_normal = self._local_attention_for_normal_dynamic(normal_features, window_sizes)

        # 3. Apply dynamic window local cross attention
        cross_refined = self._local_cross_attention_dynamic(raw_in_feat, refined_normal, window_sizes)

        # 4. Apply FFN
        ffn_refined = self._ffn_block(cross_refined)

        # 5. Final projection to required output dimension
        refined_features = self.final_proj(ffn_refined)

        return refined_features

def simple_depth_to_normal(depth_map, intrinsics):
    """
    Convert depth map to normal map.
    depth_map: [B, H, W]
    intrinsics: [B, 3, 3] camera intrinsics
    returns: [B, 3, H, W] normal map
    """
    b, h, w = depth_map.shape
    device = depth_map.device

    # Compute gradients using Sobel operator
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # Add channel dimension
    depth_map_expanded = depth_map.unsqueeze(1)  # [B, 1, H, W]

    # Compute gradients
    grad_x = F.conv2d(depth_map_expanded, sobel_x, padding=1)  # [B, 1, H, W]
    grad_y = F.conv2d(depth_map_expanded, sobel_y, padding=1)  # [B, 1, H, W]

    # Extract focal lengths
    fx = intrinsics[:, 0, 0].view(b, 1, 1, 1)  # [B, 1, 1, 1]
    fy = intrinsics[:, 1, 1].view(b, 1, 1, 1)  # [B, 1, 1, 1]

    # Compute normal vector from depth gradients
    # n = [-dz/dx * fx, -dz/dy * fy, 1]
    normal_x = -grad_x * fx
    normal_y = -grad_y * fy
    normal_z = torch.ones_like(normal_x)

    # Concatenate and normalize
    normals = torch.cat([normal_x, normal_y, normal_z], dim=1)  # [B, 3, H, W]
    normals = F.normalize(normals, dim=1)

    return normals


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor:  DepthPredictorMultiView
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderCostVolumeCfg) -> None:
        super().__init__(cfg)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in self.backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # gaussians convertor
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )
        self.gaussian_head = self.depth_predictor.to_gaussian
        feat_dim = self.depth_predictor.gau_in_channel

        # Two-stage feature refinement
        self.surfel_adaption = SurfelAdaption()
        self.feature_aggregation = FeatureAggregation(
            feature_dim=feat_dim,
            normal_dim=128,
            output_dim=feat_dim,
            hidden_dim=feat_dim
        )

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))
    
    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        #print('context["image"] shape ',context["image"].shape)
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )
        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info['images'] = rearrange(context["image"], "b v c h w -> (v b) c h w")
        extra_info["scene_names"] = scene_names
        gpp = self.cfg.gaussians_per_pixel

        depths, densities, raw_gaussians, raw_in_feat, raw_feat_channnel = self.depth_predictor(
            in_feats,
            context["intrinsics"],
            context["extrinsics"],
            context["near"],
            context["far"],
            gaussians_per_pixel=gpp,
            deterministic=deterministic,
            extra_info=extra_info,
            cnn_features=cnn_features,
        )

        depths = 200 * depths  # Remember to comment out this line when training

        # Compute dynamic window sizes
        window_sizes = self.surfel_adaption(depths, context["intrinsics"], h, w)  # [B*V, H, W]

        # Estimate normals from depths
        depths_map = rearrange(depths[:, :, :, 0, 0], "b v (h w) -> (b v) h w", h=h, w=w)
        intrinsics_bv = rearrange(context["intrinsics"], "b v i j -> (b v) i j")
        estimated_normals = simple_depth_to_normal(depths_map, intrinsics_bv)  # [B*V, 3, H, W]

        # Process estimated_normals through backbone for better feature representation
        estimated_normals_input = rearrange(estimated_normals, "(b v) c h w -> b v c h w", b=b, v=v)
        normal_features, normal_cnn_features = self.backbone(
            estimated_normals_input,
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )




        #depths = rearrange(context["depth"],"b v dpt h w -> b v (h w) dpt").unsqueeze(-1) * 1.0 / 200.0
        # print('depths max After',depths.max())
        # print('depths mean After',depths.mean())
        # print('depths min After',depths.min())
        #tensor = ddd.unsqueeze(-1) * 1.0#/ 200
        #depths = ddd
        
        # mask = (ddd >= 1) & (ddd <= 600)
        # depths = ddd.clone()  # create a copy of the original tensor
        # depths[mask] = depths[mask] * -1

        # #print('ss',depths.shape)
        # print('depths max 2',depths.max())
        # print('depths mean 2',depths.mean())
        #xxx
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
        xy_ray = xy_ray - (offset_xy - 0.5) * pixel_size
        gpp = self.cfg.gaussians_per_pixel

        b,v,r,_,cc = gaussians[..., 2:].shape
        # gaussians = self.gaussian_adapter.forward(
        #     rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
        #     rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
        #     rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
        #     depths,
        #     self.map_pdf_to_opacity(densities, global_step) / gpp,
        #     rearrange(
        #         gaussians[..., 2:],
        #         "b v r srf c -> b v r srf () c",
        #     ),
        #     (h, w),
        # )
        # Apply new feature aggregation
        normal_features = rearrange(normal_features, "b v c h w -> (b v) c h w")
        refined_features = self.feature_aggregation(
            raw_in_feat,
            normal_features,
            context,
            window_sizes,
            estimated_normals
        )

        refined_gaussians = self.gaussian_head(refined_features)
        refined_gaussians = rearrange(
            refined_gaussians, "(v b) c h w -> b v (h w) c", v=v, b=b
        )
        refined_gaussians = rearrange(
            refined_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        gaussians = self.gaussian_adapter.forward(
            rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
            rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            self.map_pdf_to_opacity(densities, global_step) / gpp,
            rearrange(
                refined_gaussians[..., 2:],
                "b v r srf c -> b v r srf () c",
            ),
            (h, w),
        )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        return Gaussians(
            rearrange(
                depths,
                "b v (h w) srf s -> b v (srf s) h w", h=h, w=w,#"b v r srf s -> b (v r srf) s",
            ),
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
            rearrange(
                gaussians.scales, "b v r srf spp xy -> b (v r srf spp) xy"
            ),
            rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            ),
        )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None
