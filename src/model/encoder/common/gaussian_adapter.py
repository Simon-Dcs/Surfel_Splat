from dataclasses import dataclass

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from ....geometry.projection import get_world_rays, homogenize_points, transform_cam2world
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance, build_scaling_rotation, build_rotation, rotation_matrix_to_quaternion, quaternion_multiply


# @dataclass
# class Gaussians:
#     means: Float[Tensor, "*batch 3"]
#     covariances: Float[Tensor, "*batch 3 3"]
#     scales: Float[Tensor, "*batch 3"]
#     rotations: Float[Tensor, "*batch 4"]
#     harmonics: Float[Tensor, "*batch 3 _"]
#     opacities: Float[Tensor, " *batch"]


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 4 4"]
    scales: Float[Tensor, "*batch 2"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]



@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
    ) -> Gaussians:
        device = extrinsics.device
        scales, rotations, sh = raw_gaussians.split((2, 4, 3 * self.d_sh), dim=-1)
        b = rotations.shape[0]
        v = rotations.shape[1]
        r = rotations.shape[2]
        srf = rotations.shape[3]
        #print('scale is ',torch.cat([scales , torch.ones_like(scales)], dim=-1).shape)
        #print('rotations',rotations.shape)
        spp = 1
        # Map scale features to valid scale range.
        scale_min = self.cfg.gaussian_scale_min
        scale_max = self.cfg.gaussian_scale_max
        scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
        h, w = image_shape
        pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
        multiplier = self.get_scale_multiplier(intrinsics, pixel_size)
        scales = scales * depths[..., None] * multiplier[..., None]
        #scales = 1.4 * scales
        #print('scale',scales.shape)
        #print('scale shape is ',scales.shape)
        #scales[..., -1] = 0.0001
        #print(scales[-1])
        # Normalize the quaternion features to yield a valid quaternion.
        # S = torch.zeros(*scales.shape[:-1], 3, 3, device=scales.device)
        # S[..., 0, 0] = scales[..., 0]  # first diagonal element
        # S[..., 1, 1] = scales[..., 1]  # second diagonal element
        # S = S.view(-1, 3, 3)
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        #N = rotations.shape[0] * rotations.shape[1] * rotations.shape[2] * rotations.shape[3]
        #temp = rotations.reshape(-1, 4)
        #print(temp.shape)
        #R = build_rotation(temp)
        #R = R.reshape(b, v, h*w, srf, 3, 3)
        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        # Create world-space covariance matrices.
        #covariances = build_scaling_rotation(scales.reshape(-1, 2), rotations.reshape(-1, 4))
        #covariances = covariances.reshape(b, v, r, srf, spp, 3, 3)
        #print('before ',covariances.shape)
        #covariances = rearrange(covariances, "(b v q srf) m n -> b v q srf m n", b=b,v=v,q=h*w,srf=srf,m=3,n=3)
        #print('extrinsics are ',extrinsics)
        c2w_rotations = extrinsics[..., :3, :3]
        #covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        
        #print('middle ',covariances.shape)
        #covariances = covariances.reshape(b, v, r, srf, 3, 3)
        # Compute Gaussian means.

        # origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        # means = origins + directions * depths[..., None]


        coordinates_homo = homogenize_points(coordinates)  # add homogeneous coordinates

        # Apply inverse of intrinsics to get normalized camera coordinates
        cam_directions = torch.einsum('...ij,...j->...i', intrinsics.inverse(), coordinates_homo)

        # In camera space, use z-depth directly
        cam_points = cam_directions * depths[..., None] / cam_directions[..., -1:]

        # Transform to world coordinates
        cam_points_homo = homogenize_points(cam_points[..., :3])
        means = transform_cam2world(cam_points_homo, extrinsics)[..., :3]





        # print('harmonics ',rotate_sh(sh, c2w_rotations[..., None, :, :]).shape)
        # print('covariances ',covariances.shape)
        # print('scales ',scales.shape)
        # print('rotations ',rotations.shape)
        ss = scales.reshape(-1,2)
        #rr = rotations.reshape(-1,4)
        #print('sss',c2w_rotations.shape)
        c2w = rotation_matrix_to_quaternion(c2w_rotations.reshape(-1,3,3))
        c2w_shape = rearrange(c2w,"(b v r srf spp) m -> b v r srf spp m",b=b,v=v,r=1,srf=srf,spp =spp,m=4)
        c2w_r = quaternion_multiply(rotations,c2w_shape).reshape(-1,4)
        xyz = means.reshape(-1,3)
        RS = build_scaling_rotation(torch.cat([ss, torch.ones_like(ss)], dim=-1), c2w_r).permute(0,2,1)
        trans = torch.zeros((ss.shape[0], 4, 4), dtype=torch.float, device=device)
        trans[:,:3,:3] = RS
        trans[:, 3,:3] = xyz
        trans[:, 3, 3] = 1
        cov = rearrange(trans,"(b v r srf spp) m n -> b v r srf spp m n",b=b,v=v,r=h*w,srf=srf,spp =spp,m=4,n=4)
        #print('means ',means.shape)
        c2w_r = rearrange(c2w_r,"(b v r srf spp) m -> b v r srf spp m",b=b,v=v,r=h*w,srf=srf,spp =spp,m=4)

        return Gaussians(
            means=means,
            covariances=cov,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: I am here to fix it!
            scales=scales,
            rotations=rotations, # .broadcast_to((*scales.shape[:-1], 4))
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 6 + 3 * self.d_sh
