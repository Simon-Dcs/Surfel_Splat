from math import isqrt
from typing import Literal

import torch
# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points
from ..encoder.costvolume.conversions import depth_to_relative_disparity


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def depths_to_points(width, height, w2v, proj, depthmap):
    c2w = (w2v.T).inverse()
    W, H = width, height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ proj
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(width, height, w2v, proj, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(width, height, w2v, proj, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 4 4"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_scales: Float[Tensor, "batch gaussian 2"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    scale_invariant: bool = False,
    use_sh: bool = True,
):
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
    device = extrinsics.device
    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        #gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    #print('Ultra problem',near,far)
    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()
    depth_ratio = 1
    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix
    # ndc2pix = torch.tensor([
    #         [w / 2, 0, 0, (w-1) / 2],
    #         [0, h / 2, 0, (h-1) / 2],
    #         [0, 0, far-near, near],
    #         [0, 0, 0, 1]]).float().cuda().T

    all_images = []
    all_radii = []
    all_depth = []
    all_normal = []
    all_ren_nor = []
    all_ren_dist = []
    #print('batch',b)
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        ndc2pix = torch.tensor([
        [w / 2, 0, 0, (w-1) / 2],
        [0, h / 2, 0, (h-1) / 2],
        [0, 0, far[i]-near[i], near[i]],
        [0, 0, 0, 1]
        ], dtype=torch.float32, device=device).T
        row, col = torch.triu_indices(3, 3)

        # world2pix =  full_projection[i] @ ndc2pix
        # gau_cov = (gaussian_covariances[i, :, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1)
        # gau_cov = rearrange(gau_cov,"g i j -> g (i j)")

        #print('gaussian_means[i] max is ',gaussian_means[i].max())
        # print("gaussian_means shape:", gaussian_means[i].shape)
        # print("mean_gradients shape:", mean_gradients.shape)
        # print("shs shape:", shs[i].shape)
        # print("gaussian_opacities shape:", gaussian_opacities[i, ..., None].shape)
        # print("gaussian_covariances max:", gaussian_covariances[i, :, row, col].max())
        # print('gaussian_covariances[i].reshape(-1,9) max ',gaussian_covariances[i].reshape(-1,9).shape)
        image, radii, allmap = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            scales = gaussian_scales[i],
            rotations = gaussian_rotations[i],
            #cov3D_precomp=gau_cov,
        )
        # print('image',image.max())
        # print('allmap',allmap.max())
        all_images.append(image)

        render_alpha = allmap[1:2]
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (view_matrix[i][:3,:3].T)).permute(2,0,1)

        #print('render_normal ',render_normal.shape)
        all_ren_nor.append(render_normal)
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
        render_depth_median = render_depth_median.squeeze(0)
        #print('render_depth',render_depth_median.shape)
        all_depth.append(render_depth_median)
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        render_dist = allmap[6:7]
        all_ren_dist.append(render_dist.squeeze(0))
        surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
        surf_normal = depth_to_normal(w,h,view_matrix[i],full_projection[i], surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        surf_normal = surf_normal * (render_alpha).detach()
        all_normal.append(surf_normal)
    # print('w',torch.stack(all_images).shape)
    # print('www',torch.stack(all_depth).shape)
    # print('wwwww',torch.stack(all_normal).shape)
    #print('IMPORTANT',torch.stack(all_images).max())
    return torch.stack(all_images), torch.stack(all_depth), torch.stack(all_normal), torch.stack(all_ren_nor), torch.stack(all_ren_dist)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 4 4"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_scales: Float[Tensor, "batch gaussian 2"],
    gaussian_rotations: Float[Tensor, "batch gaussian 4"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
):
    b, _, _ = extrinsics.shape
    h, w = image_shape
    device = extrinsics.device
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1
    depth_ratio = 1
    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    all_depth = []
    all_normal = []
    all_ren_nor = []
    all_ren_dist = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)
        ndc2pix = torch.tensor([
        [w / 2, 0, 0, (w-1) / 2],
        [0, h / 2, 0, (h-1) / 2],
        [0, 0, far[i]-near[i], near[i]],
        [0, 0, 0, 1]
        ], dtype=torch.float32, device=device).T
        row, col = torch.triu_indices(3, 3)

        # world2pix =  full_projection[i] @ ndc2pix
        # gau_cov = (gaussian_covariances[i, :, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1)
        # gau_cov = rearrange(gau_cov,"g i j -> g (i j)")
        image, radii, allmap = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            scales = gaussian_scales[i],
            rotations = gaussian_rotations[i],
            #cov3D_precomp=gau_cov,
        )
        all_images.append(image)
        #all_radii.append(radii)
        render_alpha = allmap[1:2]
        render_normal = allmap[2:5]
        all_ren_dist.append(allmap[6:7].squeeze(0))
        render_normal = (render_normal.permute(1,2,0) @ (view_matrix[i][:3,:3].T)).permute(2,0,1)
        #print('render_normal ',render_normal.shape)
        all_ren_nor.append(render_normal)
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)
        render_depth_median = render_depth_median.squeeze(0)
        #print('render_depth',render_depth_median.shape)
        all_depth.append(render_depth_median)
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        #render_dist = allmap[6:7]
        surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
        surf_normal = depth_to_normal(w,h,view_matrix[i],full_projection[i], surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        all_normal.append(surf_normal)
    return torch.stack(all_images), torch.stack(all_depth), torch.stack(all_normal), torch.stack(all_ren_nor), torch.stack(all_ren_dist)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "relative_disparity":
        fake_color = depth_to_relative_disparity(
            fake_color, near[:, None], far[:, None]
        )
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)



def render_normal(    
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
) -> Float[Tensor, "batch 3 height width"]:
    b, g, _ = gaussian_means.shape
    eigenvalues = torch.linalg.eigvals(gaussians_covariances).abs() # [batch, gaussian, 3]
    min_pos = torch.argmin(eigenvalues, dim=-1)[0] #[batch, gaussian]
    norm = torch.zeros((b, g, 3), dtype=gaussians_covariances.dtype, device=gaussians_covariances.device)
    for i in range(b):
        for j in range(g):
            norm[i][j][min_pos[i,j]] = 1
    camera_space_norm = einsum(
        extrinsics.inverse(), homogenize_points(norm), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_norm[..., :3]
    # Render using depth as color.

    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        rearrange(fake_color, "b g j -> b i j v", v = 1),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)