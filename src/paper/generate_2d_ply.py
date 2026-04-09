from pathlib import Path

import hydra
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import install_import_hook
from lightning_fabric.utilities.apply_func import apply_to_collection
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import default_collate
import json
from tqdm import tqdm
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from ..visualization.vis_depth import viz_depth_tensor
import os
from PIL import Image
import open3d as o3d

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset import get_dataset
    from src.dataset.view_sampler.view_sampler_arbitrary import ViewSamplerArbitraryCfg
    from src.geometry.projection import homogenize_points, project
    from src.global_cfg import set_cfg
    from src.misc.image_io import save_image
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.decoder.cuda_splatting import render_cuda_orthographic
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.ply_export import export_ply, export_2d_gaussian
    from src.visualization.color_map import apply_color_map_to_image
    from src.visualization.drawing.cameras import unproject_frustum_corners
    from src.visualization.drawing.lines import draw_lines
    from src.visualization.drawing.points import draw_points


with open("assets/evaluation_index_re10k.json") as f:
    scene_cfgs = json.load(f)

DEFAULT_SCENES = (
    ("scan24_train", 0, 2, 9, 6.0, [110], 1.4, 19),
)
TARGET_VIEWS = [4, 2]
OUTPUT_DATASET_NAME = os.environ.get("GENERATE_2D_PLY_OUTPUT_DATASET", "data")
sks = '40'
#paths = ['eval/pre_dtu/scan24/mask/000.png','eval/pre_dtu/scan24/mask/001.png']

FIGURE_WIDTH = 500
MARGIN = 4
GAUSSIAN_TRIM = 20
LINE_WIDTH = 1.8
LINE_COLOR = [255, 0, 0]
POINT_DENSITY = 0.5
NUM_VIEWS = 2
IMAGE_SHAPE = [256,256] # [480,640] or [256,256] or [270,480]
FLAT_TEST = False
FILTER_OUT = False

# TSDF parameters (DISABLED - TSDF mesh generation has been commented out)
# These parameters are kept for reference but are not currently used
GENERATE_MESH = False  # TSDF generation is disabled by default
TSDF_VOXEL_SIZE = 0.004  # Keep the original scale.
TSDF_SDF_TRUNC = 0.02    # Keep the original scale.
TSDF_DEPTH_TRUNC = 10.0  # Adjust to the expected depth range.
TSDF_NUM_VIEWS = -1  # -1 means use all available views


def parse_int_list(raw_value: str) -> list[int]:
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return [int(value) for value in values]


def parse_float_list(raw_value: str) -> list[float]:
    values = [value.strip() for value in raw_value.split(",") if value.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return [float(value) for value in values]


def load_scene_specs() -> tuple[tuple]:
    scene_name = os.environ.get("GENERATE_2D_PLY_SCENE_NAME")
    context_views_raw = os.environ.get("GENERATE_2D_PLY_CONTEXT_VIEWS")
    target_views_raw = os.environ.get("GENERATE_2D_PLY_TARGET_VIEWS")

    if scene_name is None or context_views_raw is None or target_views_raw is None:
        return DEFAULT_SCENES

    global TARGET_VIEWS
    TARGET_VIEWS = parse_int_list(target_views_raw)

    context_views = parse_int_list(context_views_raw)
    far = float(os.environ.get("GENERATE_2D_PLY_FAR", "6.0"))
    angles = parse_float_list(os.environ.get("GENERATE_2D_PLY_ANGLES", "110"))
    line_width = float(os.environ.get("GENERATE_2D_PLY_LINE_WIDTH", "1.4"))
    cam_div = int(os.environ.get("GENERATE_2D_PLY_CAM_DIV", "19"))

    return ((scene_name, *context_views, far, angles, line_width, cam_div),)


SCENES = load_scene_specs()

def viz_depth_tensor_2d(disp, return_numpy=False, colormap='viridis'):
    # visualize inverse depth
    #assert isinstance(disp, torch.Tensor)
    #print('disp shape',disp.shape)
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]

    return viz

def normalize_per_channel_torch(tensor):
    """
    Normalize a `[3, H, W]` tensor to `[0, 1]` per channel.

    Args:
        tensor: Tensor with shape `[3, H, W]`.

    Returns:
        The normalized tensor.
    """
    assert tensor.shape[0] == 3, "Input tensor must have 3 channels."

    normalized = torch.zeros_like(tensor, dtype=torch.float32)

    for c in range(3):
        channel = tensor[c]
        min_val = channel.min()
        max_val = channel.max()

        if max_val > min_val:
            normalized[c] = (channel - min_val) / (max_val - min_val)
        else:
            normalized[c] = channel
            
    return normalized

def vectorized_column_to_quaternion(col_tensor):
    """
    Vectorized conversion from the last rotation-matrix column to quaternions.
    """
    b, v, _, h, w = col_tensor.shape

    flat_cols = col_tensor.permute(0, 1, 3, 4, 2).reshape(-1, 3)

    R_02 = flat_cols[:, 0]  # 2 * (x*z + r*y)
    R_12 = flat_cols[:, 1]  # 2 * (y*z - r*x)
    R_22 = flat_cols[:, 2]  # 1 - 2 * (x*x + y*y)

    x2_plus_y2 = (1 - R_22) / 2
    x2_plus_y2 = torch.clamp(x2_plus_y2, min=0)  # Handle numerical error.

    z = torch.full_like(R_22, 0.5)

    x = y = torch.sqrt(x2_plus_y2 / 2)

    r = torch.zeros_like(R_22)

    y_mask = torch.abs(y) > 1e-10
    r[y_mask] = (R_02[y_mask]/2 - x[y_mask]*z[y_mask]) / y[y_mask]

    xy_mask = (~y_mask) & (torch.abs(x) > 1e-10)
    r[xy_mask] = -(R_12[xy_mask]/2 - y[xy_mask]*z[xy_mask]) / x[xy_mask]

    both_zero_mask = (~y_mask) & (~xy_mask)
    r[both_zero_mask] = 1.0

    flat_quaternions = torch.stack([r, x, y, z], dim=1)

    q_norms = torch.norm(flat_quaternions, dim=1, keepdim=True)
    q_mask = q_norms > 1e-10
    flat_quaternions[q_mask.squeeze()] = flat_quaternions[q_mask.squeeze()] / q_norms[q_mask]

    quaternions = flat_quaternions.reshape(b, v, h, w, 4).permute(0, 1, 4, 2, 3)
    
    return quaternions

def vectorized_column_to_quaternion2(col_tensor):
    """
    Vectorized conversion from the last rotation-matrix column to quaternions.
    """
    b, v, _, h, w = col_tensor.shape

    flat_cols = col_tensor.permute(0, 1, 3, 4, 2).reshape(-1, 3)

    R_02 = flat_cols[:, 0]  # 2 * (x*z + r*y)
    R_12 = flat_cols[:, 1]  # 2 * (y*z - r*x)
    R_22 = flat_cols[:, 2]  # 1 - 2 * (x*x + y*y)

    x2_plus_y2 = (1 - R_22) / 2
    x2_plus_y2 = torch.clamp(x2_plus_y2, min=0)  # Handle numerical error.

    z = torch.full_like(R_22, 0.5)

    x = y = torch.sqrt(x2_plus_y2 / 2)

    r = torch.zeros_like(R_22)

    y_mask = torch.abs(y) > 1e-10
    r[y_mask] = (R_02[y_mask]/2 - x[y_mask]*z[y_mask]) / y[y_mask]

    xy_mask = (~y_mask) & (torch.abs(x) > 1e-10)
    r[xy_mask] = -(R_12[xy_mask]/2 - y[xy_mask]*z[xy_mask]) / x[xy_mask]

    both_zero_mask = (~y_mask) & (~xy_mask)
    r[both_zero_mask] = 1.0

    flat_quaternions = torch.stack([r, x, y, z], dim=1)

    q_norms = torch.norm(flat_quaternions, dim=1, keepdim=True)

    normalized_quaternions = torch.where(
        q_norms > 1e-10,
        flat_quaternions / q_norms,
        flat_quaternions
    )

    quaternions = normalized_quaternions.reshape(b, v, h, w, 4).permute(0, 1, 4, 2, 3)
    
    return quaternions

def build_rotation(r):
    #print('rotation shape is ',r.shape)
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    #print('correct')
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

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

def build_scaling_rotation(s, r):
    #print('s',s.shape)
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = 0
    #print('R',R.shape)
    #print('L',L.shape)
    L = R @ L
    return L

def ref_view(view_list):
    diff = view_list[1] - view_list[0]
    diff = int(diff/3)
    return [view_list[0] + diff,view_list[1] - diff]

def load_mask(mask_path, target_size=(160, 128), device='cuda:0'):
    """
    Load a mask image, resize it, binarize it, and convert it to a tensor.
    """
    mask = Image.open(mask_path)
    mask = mask.resize(target_size, Image.NEAREST)
    mask_np = np.array(mask)
    binary_mask = np.where(mask_np > 0, 1, 0).astype(np.uint8)

    if len(binary_mask.shape) == 3:
        binary_mask = binary_mask.max(axis=2)
    binary_mask = np.expand_dims(binary_mask, axis=0)
    mask_tensor = torch.from_numpy(binary_mask).to(device)

    return mask_tensor

def clip_tensor(tensor, lower_bound=-70, upper_bound=70):
    """
    Set values outside `[lower_bound, upper_bound]` to zero.
    """
    mask = (tensor >= lower_bound) & (tensor <= upper_bound)
    clipped_tensor = tensor * mask
    
    return clipped_tensor

def save_depth_error_visualization_pil(depth_diff, output_path, max_error=None):
    """
    Visualize a depth error map with PIL and save it as PNG.
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(depth_diff, torch.Tensor):
        depth_diff = depth_diff.detach().cpu().numpy()

    error_map = np.abs(depth_diff)

    if max_error is None:
        max_error = np.max(error_map)

    normalized_error = np.clip(error_map / max_error, 0, 1)

    h, w = normalized_error.shape
    rgb_image = np.ones((h, w, 3), dtype=np.uint8) * 255

    rgb_image[:, :, 1] = 255 - (normalized_error * 255).astype(np.uint8)  # Green channel.
    rgb_image[:, :, 2] = 255 - (normalized_error * 255).astype(np.uint8)  # Blue channel.

    img = Image.fromarray(rgb_image)
    img.save(output_path)

    return output_path

def get_all_camera_views(cfg, scene_name, context_indices, stage="test", num_views=-1):
    """
    Extract every camera view for a scene directly from the dataset files.
    """
    dataset_temp = get_dataset(cfg.dataset, stage, None)

    if hasattr(dataset_temp, 'index') and scene_name in dataset_temp.index:
        chunk_path = dataset_temp.index[scene_name]
    else:
        chunk_path = None
        for chunk_file in dataset_temp.chunks:
            chunk_data = torch.load(chunk_file)
            for item in chunk_data:
                if item['key'] == scene_name:
                    chunk_path = chunk_file
                    break
            if chunk_path:
                break

        if chunk_path is None:
            raise ValueError(f"Scene '{scene_name}' not found in dataset")

    chunk_data = torch.load(chunk_path)
    scene_data = None
    for item in chunk_data:
        if item['key'] == scene_name:
            scene_data = item
            break

    if scene_data is None:
        raise ValueError(f"Scene '{scene_name}' not found in chunk {chunk_path}")

    cameras = scene_data['cameras']
    total_views = len(cameras)

    print(f"Scene '{scene_name}' has {total_views} total views available")

    extrinsics_all, intrinsics_all = dataset_temp.convert_poses(cameras)

    all_indices = list(range(total_views))
    target_indices = all_indices  # Use every available view, including context views.

    if num_views > 0 and len(target_indices) > num_views:
        step = len(target_indices) / num_views
        target_indices = [target_indices[int(i * step)] for i in range(num_views)]
        print(f"Sampled {num_views} views uniformly from {total_views} available views")

    print(f"Using all {len(target_indices)} views (including context) for TSDF fusion: {target_indices}")

    target_indices_tensor = torch.tensor(target_indices, dtype=torch.long)
    all_extrinsics = extrinsics_all[target_indices_tensor]  # [N, 4, 4]
    all_intrinsics = intrinsics_all[target_indices_tensor]  # [N, 3, 3]

    # Compute the scene scale used for near and far bounds.
    if len(context_indices) == 2 and cfg.dataset.make_baseline_1:
        context_extrinsics = extrinsics_all[list(context_indices)]
        a, b = context_extrinsics[:, :3, 3]
        scale = (a - b).norm()
    else:
        scale = 1.0

    return all_extrinsics, all_intrinsics, target_indices_tensor, scale

def tsdf_fusion(gaussians, decoder, all_extrinsics, all_intrinsics,
                near, far, image_shape, device,
                voxel_size=0.004, sdf_trunc=0.02, depth_trunc=6.0):
    """
    Generate a mesh from 2DGS with TSDF fusion.
    """
    print(f"\n{'='*60}")
    print("Starting TSDF Fusion for Mesh Generation")
    print(f"{'='*60}")
    print(f"Number of views: {all_extrinsics.shape[0]}")
    print(f"Image shape: {image_shape}")
    print(f"Voxel size: {voxel_size}")
    print(f"SDF truncation: {sdf_trunc}")
    print(f"Depth truncation: {depth_trunc}")
    print(f"{'='*60}\n")

    h, w = image_shape

    # 1. Create the TSDF volume.
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    # 2. Render all views.
    rgbmaps = []
    depthmaps = []

    print("Rendering all views...")
    for i in tqdm(range(all_extrinsics.shape[0]), desc="Rendering views"):
        # Render a single view.
        extrinsics = all_extrinsics[i:i+1].unsqueeze(0)  # [1, 1, 4, 4]
        intrinsics = all_intrinsics[i:i+1].unsqueeze(0)  # [1, 1, 3, 3]

        # Near and far must match the single-view dimension.
        near_single = near[:, 0:1] * 200  # [batch, 1]
        far_single = far[:, 0:1] * 200    # [batch, 1]

        output = decoder.forward(
            gaussians,
            extrinsics,
            intrinsics,
            near_single,
            far_single,
            (h, w),
            depth_mode=None,
        )

        rgbmaps.append(output.color[0, 0].cpu())  # [3, H, W]
        depth_map = output.depth[0, 0].cpu()  # [H, W]
        depthmaps.append(depth_map)

        # Print depth statistics for the first view.
        if i == 0:
            print(f"Debug - First view depth stats:")
            print(f"  Shape: {depth_map.shape}")
            print(f"  Min: {depth_map.min().item():.4f}")
            print(f"  Max: {depth_map.max().item():.4f}")
            print(f"  Mean: {depth_map.mean().item():.4f}")
            print(f"  Median: {depth_map.median().item():.4f}")
            print(f"  95th percentile: {torch.quantile(depth_map, 0.95).item():.4f}")
            print(f"  Non-zero pixels: {(depth_map > 0).sum().item()}/{depth_map.numel()}")

    # 3. Convert camera parameters to Open3D format.
    camera_params = []

    print("Converting camera parameters to Open3D format...")
    for i in range(all_extrinsics.shape[0]):
        # Build Open3D intrinsics.
        K = all_intrinsics[i].cpu().numpy()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            fx=361.54125,#float(K[0, 0]),
            fy=360.3975,#float(K[1, 1]),
            cx=82.900625,#float(K[0, 2]),
            cy=66.383875,#float(K[1, 2])
        )
        #print('I wonder',np.array(intrinsic))
        # Extrinsics are already world-to-camera (W2C) matrices.
        extrinsic = all_extrinsics[i].cpu().numpy()

        # Print the first camera extrinsic for debugging.
        if i == 0:
            print(f"Debug - First camera extrinsic (W2C):")
            print(f"  Translation: {extrinsic[:3, 3]}")
            print(f"  Rotation det: {np.linalg.det(extrinsic[:3, :3]):.4f}")

        cam = o3d.camera.PinholeCameraParameters()
        cam.intrinsic = intrinsic
        cam.extrinsic = extrinsic
        camera_params.append(cam)

    # 4. Run TSDF integration.
    print("Performing TSDF integration...")
    for i in tqdm(range(len(rgbmaps)), desc="TSDF integration"):
        rgb = rgbmaps[i]
        depth = depthmaps[i]

        # Create the RGBD image.
        rgb_np = np.asarray(
            np.clip(rgb.permute(1, 2, 0).numpy(), 0.0, 1.0) * 255,
            order="C",
            dtype=np.uint8
        )
        # Add a channel dimension to the depth map: [H, W] -> [H, W, 1].
        depth_np = np.asarray(
            depth.unsqueeze(-1).numpy(),  # [H, W] -> [H, W, 1]
            order="C",
            dtype=np.float32
        )

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_np),
            o3d.geometry.Image(depth_np),
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
            depth_scale=1.0
        )

        # Integrate into the volume.
        volume.integrate(
            rgbd,
            intrinsic=camera_params[i].intrinsic,
            extrinsic=camera_params[i].extrinsic
        )

    # 5. Extract the mesh.
    print("Extracting mesh from TSDF volume...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    print(f"Mesh extracted: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

    return mesh

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh by removing small disconnected components.
    """
    import copy

    print(f"Post-processing mesh to keep {cluster_to_keep} largest clusters...")
    mesh_clean = copy.deepcopy(mesh)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh_clean.cluster_connected_triangles()
        )

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # Determine the minimum cluster size to keep.
    total_clusters = len(cluster_n_triangles)
    if total_clusters <= cluster_to_keep:
        n_cluster = np.sort(cluster_n_triangles.copy())[0] if total_clusters > 0 else 0
    else:
        n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]

    n_cluster = max(n_cluster, 10)  # Filter out clusters with fewer than 10 triangles.

    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_clean.remove_triangles_by_mask(triangles_to_remove)
    mesh_clean.remove_unreferenced_vertices()
    mesh_clean.remove_degenerate_triangles()

    print(f"Vertices: {len(mesh.vertices)} -> {len(mesh_clean.vertices)}")
    print(f"Triangles: {len(mesh.triangles)} -> {len(mesh_clean.triangles)}")

    return mesh_clean

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def generate_point_cloud_figure(cfg_dict):
    cfg = load_typed_root_config(cfg_dict)
    #print(cfg)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg_dict.seed)
    device = torch.device("cuda:0")

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    model_wrapper = ModelWrapper.load_from_checkpoint(
        checkpoint_path,
        optimizer_cfg=cfg.optimizer,
        test_cfg=cfg.test,
        train_cfg=cfg.train,
        encoder=encoder,
        encoder_visualizer=encoder_visualizer,
        decoder=decoder,
        losses=[],
        step_tracker=None,
    )
    model_wrapper.eval()

    for idx, (scene, *context_indices, far, angles, line_width, cam_div) in enumerate(
        tqdm(SCENES)
    ):
        # Dynamically calculate the number of context and target views
        num_context_views = len(context_indices)
        num_target_views = len(TARGET_VIEWS)

        print(f"\nProcessing scene '{scene}' with {num_context_views} context views and {num_target_views} target views")
        print(f"Context indices: {context_indices}")
        print(f"Target indices: {TARGET_VIEWS}")

        scene_dir = Path("point_clouds") / OUTPUT_DATASET_NAME / f"{idx:0>6}_{scene}"
        scene_dir.mkdir(parents=True, exist_ok=True)
        # Create a dataset that always returns the desired scene.
        view_sampler_cfg = ViewSamplerArbitraryCfg(
            "arbitrary",
            num_context_views,
            num_target_views,
            context_views=list(context_indices),
            target_views=TARGET_VIEWS,
        )
        cfg.dataset.view_sampler = view_sampler_cfg
        cfg.dataset.single_scene = scene

        # Get the scene.
        dataset = get_dataset(cfg.dataset, "test", None)
        example = default_collate([next(iter(dataset))])
        example = apply_to_collection(example, Tensor, lambda x: x.to(device))
        # print('Shape is: ',example["context"]["image"].shape)
        # print('CAMERAS: ',example["context"]["intrinsics"].shape,example["context"]["extrinsics"].shape)
        # Generate the Gaussians.
        visualization_dump = {}
        # Figure out which Gaussians to mask off/throw away.
        _, _, _, h, w = example["context"]["image"].shape

        gaussians = encoder.forward(
            example["context"], False, visualization_dump=visualization_dump
        )
        gaussian_depths = rearrange(
            gaussians.depth,
            "b v c h w -> v h w (b c)",
            v=num_context_views, h=h, w=w
        )

        output = decoder.forward(
            gaussians,
            example["target"]["extrinsics"],
            example["target"]["intrinsics"],
            example["target"]["near"],
            example["target"]["far"],
            (h, w),
            depth_mode=None,)
        out_color = output.color
        out_depth = output.depth.detach().cpu().numpy()
        #out_depth = out_depth / out_depth.max()
        out_normal = -output.normal #-output.normal
        out_normal = out_normal + 1.0
        out_normal = out_normal / 2.0

        #out_normal = (out_normal - out_normal.min()) / (out_normal.max() - out_normal.min())
        out_d2n = output.d2n#(-output.d2n + 1.0) / 2.0
        #out_d2n = -output.d2n
        # print('out_d2n min ',out_d2n.min())
        # #print('out_depth',out_depth.mean())
        # print('out_normal',out_normal.min())
        # print('ww',out_normal[0,0,:,0,0])
        # print('xx',output.d2n[0,0,:,100,100])
        ref_normal = 2*example["target"]["normal"]-1
        new_rotations = vectorized_column_to_quaternion2(out_d2n)
        new_rotations = rearrange(new_rotations,"b v m h w -> b (v h w) m")
        rrr = rearrange(new_rotations,"b g m -> (b g) m")
        sss = rearrange(visualization_dump['scales']," b g m -> (b g) m")
        out_scales = rearrange(visualization_dump["scales"],"b (v h w) k -> b v k h w",v=num_context_views,h=h,w=w)
        #sss = 0.1*sss
        #FLAT_TEST = False

        #print(out_color[idx].shape)
        for v in range(num_context_views):
            context_rgb = example["context"]["image"][idx, v]  # [3, H, W]
            save_image(context_rgb, scene_dir / f"context_color_{v}.png")

        for k in range(num_target_views):
            #print('aima mean',(out_depth[idx,k] - depth[idx,k]).mean())
            #print('aima max',(out_depth[idx,k] - depth[idx,k]).min())
            #depth_diff = clip_tensor(out_depth[idx,k]-depth[idx,k,0])
            #save_depth_error_visualization_pil(depth_diff,f"{base}_depth_{k}.png")
            #save_depth_error_visualization_pil(out_depth[idx,k],f"{base}_depth_{k}.png")
            #save_image(-2 * example["target"]["normal"][idx, k] + 1, scene_dir / f"real_normal_{k}.png")
            save_image(out_color[idx, k], scene_dir / f"color_{k}.png")
            #save_image(ref_normal[idx,k], f"{base}_ref_normal_{k}.png")
            #save_image(out_depth[idx,k]/1000, f"{base}_depth_{k}.png")
            #save_image(clip_tensor(out_depth[idx,k]/1000,0,600), f"{base}_depth_{k}.png")
            #out_depth2 = viz_depth_tensor_2d(1.0 / (out_depth[idx,k]+ 0.001), return_numpy=True)
            out_depth2 = viz_depth_tensor_2d(out_depth[idx,k], return_numpy=True)
            Image.fromarray(out_depth2).save(scene_dir / f"depth_{k}.png")
            #save_image(normalize_per_channel_torch(out_normal[idx,k]), f"{base}_normal_{k}.png")
            #kkk = 1 - out_normal[idx,k] # [[0,2,1],:,:]
            #jjj = out_d2n[idx,k]
            #save_image(out_normal[idx, k], scene_dir / f"normal_{k}.png")
            #np.save(f'./temp2/depth'+ sks +f'_{k}.npy',depth[idx,k,0].detach().cpu().numpy())
            #np.save(f'./temp2/scales'+ sks +f'_{k}.npy',out_scales[idx,k].detach().cpu().numpy())
            #np.save(f'./gs2mesh/normal_{k}.npy',out_normal[idx,k].detach().cpu().numpy())
            #save_image(out_d2n[idx, k], scene_dir / f"d2n_{k}.png")
            #np.save(f'./gs2mesh/d2n_{k}.npy',out_d2n[idx,k].detach().cpu().numpy())
        #np.save('./gs2mesh/opacity.npy',gaussians.opacities[0].detach().cpu().numpy())



        if FILTER_OUT:
            means = rearrange(
                gaussians.means[0], "(v h w) xyz -> h w v xyz", v=num_context_views, h=h, w=w
            )
            scales = rearrange(
                visualization_dump["scales"][0], "(v h w) xyz -> h w v xyz",  v=num_context_views, h=h, w=w
            )
            rotations = rearrange(
                visualization_dump["rotations"][0], "(v h w) xyz -> h w v xyz",  v=num_context_views, h=h, w=w
            )
            harmonics = rearrange(
                gaussians.harmonics[0],   "(v h w) xyz sh -> h w v xyz sh",  v=num_context_views, h=h, w=w
            )
            opa = rearrange(
                gaussians.opacities[0], "(v h w) -> h w v",  v=num_context_views, h=h, w=w
            )
            print('means.shape',means.shape)
            means_list = []
            scales_list = []
            rotations_list = []
            harmonics_list = []
            opa_list = []
            for k in range(num_context_views):
                mask = load_mask(paths[k],(h,w),device)
                #mask = masks[0]
                indices = torch.nonzero(mask, as_tuple=True)
                means_list.append(means[:,:,k][indices[0], indices[1]])
                scales_list.append(scales[:,:,k][indices[0], indices[1]])
                rotations_list.append(rotations[:,:,k][indices[0], indices[1]])
                harmonics_list.append(harmonics[:,:,k][indices[0], indices[1]])
                opa_list.append(opa[:,:,k][indices[0], indices[1]])

            #print('hhh',mask1.shape)
            means = torch.cat(means_list,dim=0)
            print('hhh',means.shape)
            scales = torch.cat(scales_list,dim=0)
            rotations = torch.cat(rotations_list,dim=0)
            harmonics = torch.cat(harmonics_list,dim=0)
            opa = torch.cat(opa_list,dim=0)
            export_2d_gaussian(
                example["context"]["extrinsics"][0, 0],
                means,
                scales,
                rotations,
                #trim(new_rotations)[0],
                harmonics,
                opa,
                scene_dir / "gaussians.ply",
            )
        else:
            means = rearrange(
                gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=num_context_views, h=h, w=w
            )
            #print(means.shape)
            means = homogenize_points(means)
            w2c = example["context"]["extrinsics"].inverse()[0]
            means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3]
            #print(means.shape)
            # Create a mask to filter the Gaussians. First, throw away Gaussians at the
            # borders, since they're generally of lower quality.
            mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
            mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1
            #print(mask.shape)
            # Then, drop Gaussians that are really far away.
            mask = mask & (means[..., 2] < far * 200)
            #print(mask.shape)
            def trim(element):
                element = rearrange(
                    element, "() (v h w spp) ... -> h w spp v ...", v=num_context_views, h=h, w=w
                )
                return element[mask][None]

            export_2d_gaussian(
                example["context"]["extrinsics"][0, 0],
                trim(gaussians.means)[0],
                trim(visualization_dump["scales"])[0],
                trim(visualization_dump["rotations"])[0],
                #trim(new_rotations)[0],
                trim(gaussians.harmonics)[0],
                trim(gaussians.opacities)[0],
                scene_dir / "gaussians.ply",
            )

        # TSDF mesh generation has been disabled
        # The code below is commented out to focus on multi-view 2DGS rendering

        # # Save camera parameters for TSDF mesh generation
        camera_params = {
            'context_intrinsics': example["context"]["intrinsics"].cpu().numpy().tolist(),
            'context_extrinsics': example["context"]["extrinsics"].cpu().numpy().tolist(),
            'target_intrinsics': example["target"]["intrinsics"].cpu().numpy().tolist(),
            'target_extrinsics': example["target"]["extrinsics"].cpu().numpy().tolist(),
            'near': float(example["context"]["near"][0, 0].cpu().numpy()),
            'far': float(example["context"]["far"][0, 0].cpu().numpy()),
            'image_shape': [h, w],
            'scene': scene,
            'context_indices': list(context_indices),
            'target_indices': TARGET_VIEWS,
        }
        with open(scene_dir / "cameras.json", 'w') as f:
            json.dump(camera_params, f, indent=2)
        print(f"Saved camera parameters to {scene_dir / 'cameras.json'}")

        # # TSDF Mesh Generation
        # if GENERATE_MESH:
        #     try:
        #         print(f"\n{'='*60}")
        #         print(f"Generating TSDF mesh for scene: {scene}")
        #         print(f"{'='*60}\n")

        #         # Get all camera views from dataset
        #         print("Extracting all camera views from dataset...")
        #         all_extrinsics, all_intrinsics, all_indices, scale = get_all_camera_views(
        #             cfg, scene, context_indices, stage="test", num_views=TSDF_NUM_VIEWS
        #         )
        #         print(f"Found {all_extrinsics.shape[0]} camera views")
        #         #print(all_extrinsics)
        #         print(f"View indices: {all_indices.tolist()}")

        #         # Move to device
        #         all_extrinsics = all_extrinsics.to(device)
        #         all_intrinsics = all_intrinsics.to(device)

        #         # Perform TSDF fusion
        #         mesh = tsdf_fusion(
        #             gaussians=gaussians,
        #             decoder=decoder,
        #             all_extrinsics=all_extrinsics,
        #             all_intrinsics=all_intrinsics,
        #             near=example["context"]["near"],
        #             far=example["context"]["far"],
        #             image_shape=(h, w),
        #             device=device,
        #             voxel_size=TSDF_VOXEL_SIZE,
        #             sdf_trunc=TSDF_SDF_TRUNC,
        #             depth_trunc=TSDF_DEPTH_TRUNC,
        #         )

        #         # Save raw mesh
        #         mesh_path = base / "mesh_tsdf.ply"
        #         o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        #         print(f"\n✓ Saved TSDF mesh to {mesh_path}")

        #         # Post-process and save cleaned mesh
        #         print("\nPost-processing mesh...")
        #         mesh_post = post_process_mesh(mesh, cluster_to_keep=1000)
        #         mesh_post_path = base / "mesh_tsdf_post.ply"
        #         o3d.io.write_triangle_mesh(str(mesh_post_path), mesh_post)
        #         print(f"✓ Saved post-processed mesh to {mesh_post_path}")

        #         # Save TSDF metadata
        #         tsdf_metadata = {
        #             'voxel_size': TSDF_VOXEL_SIZE,
        #             'sdf_trunc': TSDF_SDF_TRUNC,
        #             'depth_trunc': TSDF_DEPTH_TRUNC,
        #             'num_views': all_extrinsics.shape[0],
        #             'view_indices': all_indices.tolist(),
        #             'raw_vertices': len(mesh.vertices),
        #             'raw_triangles': len(mesh.triangles),
        #             'post_vertices': len(mesh_post.vertices),
        #             'post_triangles': len(mesh_post.triangles),
        #         }
        #         with open(base / "tsdf_metadata.json", 'w') as f:
        #             json.dump(tsdf_metadata, f, indent=2)
        #         print(f"✓ Saved TSDF metadata to {base / 'tsdf_metadata.json'}")

        #         print(f"\n{'='*60}")
        #         print(f"TSDF mesh generation completed for {scene}")
        #         print(f"{'='*60}\n")

        #     except Exception as e:
        #         print(f"\n{'!'*60}")
        #         print(f"Error during TSDF mesh generation: {e}")
        #         print(f"{'!'*60}\n")
        #         import traceback
        #         traceback.print_exc()

        # Transform means into camera space.


        # print('max 1 x ',gaussians.means[:,:,0].mean())
        # print('max 1 y ',gaussians.means[:,:,1].mean())
        # print('max 1 z ',gaussians.means[:,:,2].mean())
        # print('aaa',means.shape)
        # means = homogenize_points(means)
        # print('bbb',means.shape)
        # w2c = example["context"]["extrinsics"].inverse()[0]

        # means = einsum(w2c, means, "v i j, ... v j -> ... v i")[..., :3]
        # print('ccc',means.shape)
        # Create a mask to filter the Gaussians. First, throw away Gaussians at the
        # borders, since they're generally of lower quality.

        # mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
        # mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

        # Then, drop Gaussians that are really far away.
        #mask = mask & (means[..., 2] < far * 2)
        #print('mask',mask.shape)
        # def trim(element):
        #     element = rearrange(
        #         element, "() (v h w spp) ... -> h w spp v ...", v=NUM_VIEWS, h=h, w=w
        #     )
        #     return element[mask][None]
        # print('guide',gaussians.means.shape)
        # print('guide',visualization_dump["scales"].shape)
        # print('guide',visualization_dump["rotations"].shape)
        # print('guide',gaussians.harmonics.shape)
        # print('guide',gaussians.opacities.shape)
        # export_2d_gaussian(
        #     example["context"]["extrinsics"][0, 0],
        #     trim(gaussians.means)[0],
        #     trim(visualization_dump["scales"])[0],
        #     trim(visualization_dump["rotations"])[0],
        #     #trim(new_rotations)[0],
        #     trim(gaussians.harmonics)[0],
        #     trim(gaussians.opacities)[0],
        #     base / "gaussians.ply",
        # )
        
        a = 1
    a = 1


if __name__ == "__main__":
    with torch.no_grad():
        generate_point_cloud_figure()
