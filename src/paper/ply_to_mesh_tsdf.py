"""
Convert 2D Gaussian Splatting PLY to mesh using TSDF volume integration.

This script loads a PLY file containing 2DGS parameters and camera parameters,
renders RGB-D images from multiple viewpoints, and fuses them into a mesh using TSDF.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from ply_loader import load_gaussians_from_ply, Gaussians2D


def load_camera_params(camera_json_path: Path) -> dict:
    """Load camera parameters from JSON file."""
    with open(camera_json_path, 'r') as f:
        params = json.load(f)
    print(f"Loaded camera parameters from {camera_json_path}")
    print(f"  Scene: {params['scene']}")
    print(f"  Image shape: {params['image_shape']}")
    print(f"  Near: {params['near']}, Far: {params['far']}")
    print(f"  Context views: {params['context_indices']}")
    return params


def create_camera_intrinsics(intrinsics_matrix: np.ndarray, image_shape: List[int]) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Create Open3D camera intrinsics from matrix.

    Args:
        intrinsics_matrix: [3, 3] intrinsics matrix
        image_shape: [height, width]

    Returns:
        Open3D PinholeCameraIntrinsic object
    """
    h, w = image_shape
    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]

    return o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )


def render_2dgs_view(
    gaussians: Gaussians2D,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    image_shape: Tuple[int, int],
    near: float,
    far: float,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Render RGB and depth from 2D Gaussians using the project's decoder.

    Args:
        gaussians: 2D Gaussian parameters
        extrinsics: [4, 4] camera extrinsics (world to camera)
        intrinsics: [3, 3] camera intrinsics
        image_shape: (height, width)
        near: near plane
        far: far plane
        device: device to use

    Returns:
        rgb: [H, W, 3] RGB image (0-1 range)
        depth: [H, W] depth map
    """
    # Import decoder here to avoid circular imports
    from jaxtyping import install_import_hook
    with install_import_hook(("src",), ("beartype", "beartype")):
        from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg

    h, w = image_shape

    # Prepare Gaussian parameters in the format expected by decoder
    # The decoder expects batch format
    means = gaussians.means[None, ...]  # [1, N, 3]

    # Build covariance matrices for 2D Gaussians
    from ply_loader import build_scaling_rotation
    covariances = build_scaling_rotation(gaussians.scales, gaussians.rotations)
    # Add homogeneous coordinate
    cov_4x4 = torch.zeros((covariances.shape[0], 4, 4), device=device, dtype=torch.float32)
    cov_4x4[:, :3, :3] = covariances
    cov_4x4[:, 3, :3] = gaussians.means
    cov_4x4[:, 3, 3] = 1
    covariances = cov_4x4[None, ...]  # [1, N, 4, 4]

    opacities = gaussians.opacities.T[None, ...]  # [1, 1, N]

    # Combine DC and rest of SH coefficients
    if gaussians.features_rest.shape[1] > 0:
        harmonics = torch.cat([
            gaussians.features_dc[:, None, :],  # [N, 1, 3]
            gaussians.features_rest  # [N, K, 3]
        ], dim=1)
    else:
        harmonics = gaussians.features_dc[:, None, :]  # [N, 1, 3]
    harmonics = harmonics[None, ...]  # [1, N, K, 3]

    # Prepare camera parameters
    extrinsics_tensor = torch.from_numpy(extrinsics).float().to(device)[None, None, ...]  # [1, 1, 4, 4]
    intrinsics_tensor = torch.from_numpy(intrinsics).float().to(device)[None, None, ...]  # [1, 1, 3, 3]
    near_tensor = torch.tensor([[near]], device=device, dtype=torch.float32)
    far_tensor = torch.tensor([[far]], device=device, dtype=torch.float32)

    # Create a minimal Gaussians object for decoder
    from dataclasses import dataclass
    @dataclass
    class GaussiansForDecoder:
        means: torch.Tensor
        covariances: torch.Tensor
        harmonics: torch.Tensor
        opacities: torch.Tensor

    gaussians_for_decoder = GaussiansForDecoder(
        means=means,
        covariances=covariances,
        harmonics=harmonics,
        opacities=opacities
    )

    # Initialize decoder
    decoder_cfg = DecoderSplattingCUDACfg()
    decoder = DecoderSplattingCUDA(decoder_cfg)
    decoder = decoder.to(device)
    decoder.eval()

    # Render
    with torch.no_grad():
        output = decoder.forward(
            gaussians_for_decoder,
            extrinsics_tensor,
            intrinsics_tensor,
            near_tensor,
            far_tensor,
            (h, w),
            depth_mode=None
        )

    # Extract RGB and depth
    rgb = output.color[0, 0].permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
    depth = output.depth[0, 0, 0].cpu().numpy()  # [H, W]

    # Clip RGB to [0, 1]
    rgb = np.clip(rgb, 0, 1)

    return rgb, depth


def tsdf_fusion(
    rgb_images: List[np.ndarray],
    depth_images: List[np.ndarray],
    camera_intrinsics: List[o3d.camera.PinholeCameraIntrinsic],
    camera_extrinsics: List[np.ndarray],
    voxel_size: float = 0.004,
    sdf_trunc: float = 0.02,
    depth_trunc: float = 3.0
) -> o3d.geometry.TriangleMesh:
    """
    Perform TSDF volume integration to create mesh.

    Args:
        rgb_images: List of RGB images [H, W, 3] in range [0, 1]
        depth_images: List of depth maps [H, W]
        camera_intrinsics: List of Open3D camera intrinsics
        camera_extrinsics: List of [4, 4] extrinsics matrices
        voxel_size: Voxel size for TSDF volume
        sdf_trunc: Truncation distance for SDF
        depth_trunc: Maximum depth to consider

    Returns:
        Reconstructed mesh
    """
    print("\nRunning TSDF volume integration...")
    print(f"  voxel_size: {voxel_size}")
    print(f"  sdf_trunc: {sdf_trunc}")
    print(f"  depth_trunc: {depth_trunc}")

    # Create TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # Integrate all views
    for i, (rgb, depth, intrinsic, extrinsic) in enumerate(
        tqdm(
            zip(rgb_images, depth_images, camera_intrinsics, camera_extrinsics),
            total=len(rgb_images),
            desc="TSDF integration"
        )
    ):
        # Convert to Open3D format
        rgb_o3d = o3d.geometry.Image(
            np.asarray(np.clip(rgb, 0.0, 1.0) * 255, order="C", dtype=np.uint8)
        )
        depth_o3d = o3d.geometry.Image(
            np.asarray(depth, order="C", dtype=np.float32)
        )

        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0
        )

        # Integrate into volume
        volume.integrate(rgbd, intrinsic=intrinsic, extrinsic=extrinsic)

    # Extract mesh
    print("Extracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"Extracted mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")

    return mesh


def post_process_mesh(mesh: o3d.geometry.TriangleMesh, cluster_to_keep: int = 50) -> o3d.geometry.TriangleMesh:
    """
    Post-process mesh to remove small disconnected components.

    Args:
        mesh: Input mesh
        cluster_to_keep: Number of largest clusters to keep

    Returns:
        Cleaned mesh
    """
    print(f"\nPost-processing mesh (keeping {cluster_to_keep} largest clusters)...")

    import copy
    mesh_clean = copy.deepcopy(mesh)

    # Cluster connected triangles
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = mesh_clean.cluster_connected_triangles()

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    # Determine threshold
    total_clusters = len(cluster_n_triangles)
    if total_clusters <= cluster_to_keep:
        n_cluster = np.sort(cluster_n_triangles.copy())[0] if total_clusters > 0 else 0
    else:
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]

    n_cluster = max(n_cluster, 50)  # Filter meshes smaller than 50 triangles

    # Remove small clusters
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_clean.remove_triangles_by_mask(triangles_to_remove)
    mesh_clean.remove_unreferenced_vertices()
    mesh_clean.remove_degenerate_triangles()

    print(f"  Vertices: {len(mesh.vertices)} -> {len(mesh_clean.vertices)}")
    print(f"  Triangles: {len(mesh.triangles)} -> {len(mesh_clean.triangles)}")

    return mesh_clean


def estimate_bounding_sphere(camera_extrinsics: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Estimate scene bounding sphere from camera positions.

    Args:
        camera_extrinsics: List of [4, 4] extrinsics matrices (world to camera)

    Returns:
        center: [3] center of bounding sphere
        radius: radius of bounding sphere
    """
    # Convert to camera-to-world
    c2ws = [np.linalg.inv(ext) for ext in camera_extrinsics]
    camera_positions = np.array([c2w[:3, 3] for c2w in c2ws])

    # Estimate center using mean of camera positions
    center = np.mean(camera_positions, axis=0)

    # Estimate radius as minimum distance from center to any camera
    distances = np.linalg.norm(camera_positions - center, axis=1)
    radius = np.min(distances)

    print(f"\nEstimated bounding sphere:")
    print(f"  Center: {center}")
    print(f"  Radius: {radius:.2f}")

    return center, radius


def main():
    parser = argparse.ArgumentParser(description="Convert 2DGS PLY to mesh using TSDF")
    parser.add_argument("--ply_dir", type=str, required=True,
                        help="Directory containing gaussians.ply and cameras.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output mesh path (default: <ply_dir>/mesh.ply)")
    parser.add_argument("--voxel_size", type=float, default=-1,
                        help="Voxel size for TSDF (default: auto from depth_trunc/1024)")
    parser.add_argument("--sdf_trunc", type=float, default=-1,
                        help="SDF truncation distance (default: 5.0 * voxel_size)")
    parser.add_argument("--depth_trunc", type=float, default=-1,
                        help="Maximum depth (default: 2.0 * radius)")
    parser.add_argument("--mesh_res", type=int, default=1024,
                        help="Mesh resolution for auto voxel_size calculation")
    parser.add_argument("--num_cluster", type=int, default=50,
                        help="Number of largest clusters to keep in post-processing")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--use_all_views", action="store_true",
                        help="Use all available views (context + target) for TSDF fusion")

    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    ply_path = ply_dir / "gaussians.ply"
    camera_json_path = ply_dir / "cameras.json"

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    if not camera_json_path.exists():
        raise FileNotFoundError(f"Camera parameters not found: {camera_json_path}")

    # Set output path
    if args.output is None:
        output_path = ply_dir / "mesh.ply"
    else:
        output_path = Path(args.output)

    print("=" * 80)
    print("2DGS to Mesh Conversion (TSDF Method)")
    print("=" * 80)

    # Load 2D Gaussians
    gaussians = load_gaussians_from_ply(ply_path, device=args.device)

    # Load camera parameters
    camera_params = load_camera_params(camera_json_path)
    image_shape = camera_params['image_shape']
    near = camera_params['near']
    far = camera_params['far']

    # Prepare camera lists
    if args.use_all_views:
        # Use both context and target views
        intrinsics_list = (
            camera_params['context_intrinsics'] +
            camera_params['target_intrinsics']
        )
        extrinsics_list = (
            camera_params['context_extrinsics'] +
            camera_params['target_extrinsics']
        )
        print(f"\nUsing all {len(intrinsics_list)} views for TSDF fusion")
    else:
        # Use only context views
        intrinsics_list = camera_params['context_intrinsics']
        extrinsics_list = camera_params['context_extrinsics']
        print(f"\nUsing {len(intrinsics_list)} context views for TSDF fusion")

    # Convert to numpy arrays
    intrinsics_np = [np.array(intr) for intr in intrinsics_list]
    extrinsics_np = [np.array(extr) for extr in extrinsics_list]

    # Estimate bounding sphere for auto parameter calculation
    center, radius = estimate_bounding_sphere(extrinsics_np)

    # Set TSDF parameters
    if args.depth_trunc < 0:
        depth_trunc = radius * 2.0
    else:
        depth_trunc = args.depth_trunc

    if args.voxel_size < 0:
        voxel_size = depth_trunc / args.mesh_res
    else:
        voxel_size = args.voxel_size

    if args.sdf_trunc < 0:
        sdf_trunc = 5.0 * voxel_size
    else:
        sdf_trunc = args.sdf_trunc

    print(f"\nTSDF parameters:")
    print(f"  depth_trunc: {depth_trunc:.4f}")
    print(f"  voxel_size: {voxel_size:.6f}")
    print(f"  sdf_trunc: {sdf_trunc:.6f}")

    # Render views
    print(f"\nRendering {len(intrinsics_np)} views...")
    rgb_images = []
    depth_images = []
    camera_intrinsics_o3d = []

    for i, (intr, extr) in enumerate(tqdm(
        zip(intrinsics_np, extrinsics_np),
        total=len(intrinsics_np),
        desc="Rendering views"
    )):
        rgb, depth = render_2dgs_view(
            gaussians,
            extr,
            intr,
            tuple(image_shape),
            near,
            far,
            device=args.device
        )
        rgb_images.append(rgb)
        depth_images.append(depth)
        camera_intrinsics_o3d.append(create_camera_intrinsics(intr, image_shape))

    # TSDF fusion
    mesh = tsdf_fusion(
        rgb_images,
        depth_images,
        camera_intrinsics_o3d,
        extrinsics_np,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        depth_trunc=depth_trunc
    )

    # Post-process
    mesh_clean = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)

    # Save mesh
    print(f"\nSaving mesh to {output_path}...")
    o3d.io.write_triangle_mesh(str(output_path), mesh_clean)

    # Also save unprocessed mesh
    output_path_raw = output_path.parent / (output_path.stem + "_raw" + output_path.suffix)
    o3d.io.write_triangle_mesh(str(output_path_raw), mesh)
    print(f"Saved raw mesh to {output_path_raw}")

    print("\n" + "=" * 80)
    print("Mesh generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
