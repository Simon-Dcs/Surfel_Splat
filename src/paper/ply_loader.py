"""
Load 2D Gaussian Splatting parameters from PLY file.
"""
import numpy as np
import torch
from plyfile import PlyData
from pathlib import Path
from typing import Tuple
from dataclasses import dataclass


@dataclass
class Gaussians2D:
    """Container for 2D Gaussian Splatting parameters."""
    means: torch.Tensor  # [N, 3]
    scales: torch.Tensor  # [N, 2]
    rotations: torch.Tensor  # [N, 4] quaternions
    opacities: torch.Tensor  # [N, 1]
    features_dc: torch.Tensor  # [N, 3]
    features_rest: torch.Tensor  # [N, K, 3] where K is number of SH coefficients - 1


def load_gaussians_from_ply(
    ply_path: Path,
    device: str = "cuda"
) -> Gaussians2D:
    """
    Load 2D Gaussian Splatting parameters from PLY file.

    Args:
        ply_path: Path to the PLY file
        device: Device to load tensors to

    Returns:
        Gaussians2D object containing all parameters
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    print(f"Loading 2DGS from {ply_path}...")
    plydata = PlyData.read(str(ply_path))
    vertex_data = plydata['vertex']

    # Extract positions
    means = np.stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ], axis=1).astype(np.float32)

    # Extract scales (2D Gaussians have 2 scales)
    scales = np.stack([
        vertex_data['scale_0'],
        vertex_data['scale_1']
    ], axis=1).astype(np.float32)

    # Extract rotations (quaternions: w, x, y, z)
    rotations = np.stack([
        vertex_data['rot_0'],
        vertex_data['rot_1'],
        vertex_data['rot_2'],
        vertex_data['rot_3']
    ], axis=1).astype(np.float32)

    # Extract opacities
    opacities = vertex_data['opacity'].astype(np.float32)[:, np.newaxis]

    # Extract DC component of spherical harmonics (RGB color)
    features_dc = np.stack([
        vertex_data['f_dc_0'],
        vertex_data['f_dc_1'],
        vertex_data['f_dc_2']
    ], axis=1).astype(np.float32)

    # Extract rest of spherical harmonics if present
    # Count how many f_rest_* fields exist
    extra_f_names = [name for name in vertex_data.data.dtype.names if name.startswith('f_rest_')]
    if len(extra_f_names) > 0:
        # Assuming format: f_rest_0, f_rest_1, ..., f_rest_N
        # where N = (num_sh_coeffs - 1) * 3 - 1
        num_extra_features = len(extra_f_names)
        features_rest_flat = np.stack([
            vertex_data[name] for name in sorted(extra_f_names)
        ], axis=1).astype(np.float32)

        # Reshape to [N, K, 3] where K = num_sh_coeffs - 1
        num_coeffs = (num_extra_features // 3)
        features_rest = features_rest_flat.reshape(-1, num_coeffs, 3)
    else:
        # No extra SH coefficients
        features_rest = np.zeros((means.shape[0], 0, 3), dtype=np.float32)

    # Convert to torch tensors
    gaussians = Gaussians2D(
        means=torch.from_numpy(means).to(device),
        scales=torch.from_numpy(scales).to(device),
        rotations=torch.from_numpy(rotations).to(device),
        opacities=torch.from_numpy(opacities).to(device),
        features_dc=torch.from_numpy(features_dc).to(device),
        features_rest=torch.from_numpy(features_rest).to(device)
    )

    print(f"Loaded {means.shape[0]} 2D Gaussians")
    print(f"  - Means: {gaussians.means.shape}")
    print(f"  - Scales: {gaussians.scales.shape}")
    print(f"  - Rotations: {gaussians.rotations.shape}")
    print(f"  - Opacities: {gaussians.opacities.shape}")
    print(f"  - Features DC: {gaussians.features_dc.shape}")
    print(f"  - Features Rest: {gaussians.features_rest.shape}")

    return gaussians


def build_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrices from quaternions.

    Args:
        quaternions: [N, 4] tensor of quaternions (w, x, y, z)

    Returns:
        [N, 3, 3] rotation matrices
    """
    # Normalize quaternions
    norm = torch.sqrt(
        quaternions[:, 0]**2 + quaternions[:, 1]**2 +
        quaternions[:, 2]**2 + quaternions[:, 3]**2
    )
    q = quaternions / norm[:, None]

    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros((q.size(0), 3, 3), device=quaternions.device, dtype=quaternions.dtype)

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    return R


def build_scaling_rotation(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """
    Build covariance matrices for 2D Gaussians.

    Args:
        scales: [N, 2] scale parameters
        rotations: [N, 4] quaternions

    Returns:
        [N, 3, 3] covariance matrices (with z-scale = 0 for 2D Gaussians)
    """
    L = torch.zeros((scales.shape[0], 3, 3), dtype=torch.float32, device=scales.device)
    R = build_rotation_matrix(rotations)

    L[:, 0, 0] = scales[:, 0]
    L[:, 1, 1] = scales[:, 1]
    L[:, 2, 2] = 0  # 2D Gaussian, no depth extent

    # Covariance = R @ L @ L^T @ R^T
    return R @ L


if __name__ == "__main__":
    # Test loading
    ply_path = Path("point_clouds/re10k/000000_scan24_train/gaussians.ply")
    gaussians = load_gaussians_from_ply(ply_path)
    print("\nSuccessfully loaded 2D Gaussians!")
