import subprocess
from pathlib import Path
from typing import Literal, TypedDict, Optional

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
import argparse
from tqdm import tqdm
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input dtu raw directory")
parser.add_argument("--output_dir", type=str, help="output directory")
parser.add_argument("--depth_dir", type=str, help="depth maps directory", default=None)
parser.add_argument("--normal_dir", type=str, help="normal maps directory", default=None)
args = parser.parse_args()

INPUT_IMAGE_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)
DEPTH_DIR = Path(args.depth_dir) if args.depth_dir else None
NORMAL_DIR = Path(args.normal_dir) if args.normal_dir else None

# Target 100 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(1e8)


def build_camera_info(id_list, root_dir):
    """Return the camera information for the given id_list"""
    intrinsics, world2cams, cam2worlds, near_fars = {}, {}, {}, {}
    scale_factor = 1.0 / 200
    downSample = 1.0
    for vid in id_list:
        proj_mat_filename = os.path.join(
            root_dir, f"Cameras/train/{vid:08d}_cam.txt")
        intrinsic, extrinsic, near_far = read_cam_file(proj_mat_filename)

        # intrinsic[:2] *= 4
        # intrinsic[:2] = intrinsic[:2] * downSample
        intrinsics[vid] = intrinsic

        extrinsic[:3, 3] *= scale_factor
        world2cams[vid] = extrinsic
        cam2worlds[vid] = np.linalg.inv(extrinsic)

        near_fars[vid] = near_far

    return intrinsics, world2cams, cam2worlds, near_fars


def read_cam_file(filename):
    scale_factor = 1.0 / 200

    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsic = np.fromstring(" ".join(lines[1:5]), dtype=np.float32, sep=" ")
    extrinsic = extrinsic.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsic = np.fromstring(" ".join(lines[7:10]), dtype=np.float32, sep=" ")
    intrinsic = intrinsic.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0]) * scale_factor
    depth_max = depth_min + float(lines[11].split()[1]) * 192 * scale_factor
    near_far = [depth_min, depth_max]
    return intrinsic, extrinsic, near_far


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    """Dynamically find all existing scan directories"""
    rectified_dir = INPUT_IMAGE_DIR / "Rectified"
    existing_keys = []
 
    RANGING_LIST = [24,37 ,40 ,55 ,63 ,65 ,69, 83, 97 ,105 ,106, 110 ,114 ,118, 122]
    for i in RANGING_LIST:
        key = f"scan{i}_train"
        if (rectified_dir / key).exists():
            existing_keys.append(key)
    
    print(f"Found {len(existing_keys)} existing keys: {existing_keys}")
    return existing_keys


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""
    images_dict = {}
    for cur_id in range(1, 50):
        cur_image_name = f"rect_{cur_id:03d}_3_r5000.png"
        img_path = example_path / cur_image_name
        if img_path.exists():
            img_bin = load_raw(img_path)
            images_dict[cur_id - 1] = img_bin
        else:
            print(f"Warning: Image {img_path} not found")

    return images_dict


def load_depths(scan_key: str, timestamps: list[int]) -> Optional[dict[int, UInt8[Tensor, "..."]]]:
    """Load depth maps as raw bytes."""
    if DEPTH_DIR is None:
        return None
    
    depths_dict = {}
    scan_depth_dir = DEPTH_DIR / scan_key
    
    if not scan_depth_dir.exists():
        print(f"Warning: Depth directory {scan_depth_dir} not found")
        return None
    
    for idx, timestamp in enumerate(timestamps):
        depth_name = f"depth_map_{idx:04d}.png"
        depth_path = scan_depth_dir / depth_name
        
        if depth_path.exists():
            depth_bin = load_raw(depth_path)
            depths_dict[timestamp] = depth_bin
        else:
            print(f"Warning: Depth map {depth_path} not found")
    
    if not depths_dict:
        return None
        
    return depths_dict


def load_normals(scan_key: str, timestamps: list[int]) -> Optional[dict[int, UInt8[Tensor, "..."]]]:
    """Load normal maps as raw bytes."""
    if NORMAL_DIR is None:
        return None
    
    normals_dict = {}
    scan_normal_dir = NORMAL_DIR / scan_key
    
    if not scan_normal_dir.exists():
        print(f"Warning: Normal directory {scan_normal_dir} not found")
        return None
    
    for idx, timestamp in enumerate(timestamps):
        #idxx = idx + 1
        normal_name = f"normal_map_{idx:04d}.png"
        normal_path = scan_normal_dir / normal_name
        
        if normal_path.exists():
            normal_bin = load_raw(normal_path)
            normals_dict[timestamp] = normal_bin
        else:
            print(f"Warning: Normal map {normal_path} not found")
    
    if not normals_dict:
        return None
        
    return normals_dict


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]
    depths: Optional[list[UInt8[Tensor, "..."]]]
    normals: Optional[list[UInt8[Tensor, "..."]]]


def load_metadata(intrinsics, world2cams) -> Metadata:
    timestamps = []
    cameras = []
    url = ""

    for vid, intr in intrinsics.items():
        timestamps.append(int(vid))

        # normalized the intr
        fx = intr[0, 0]
        fy = intr[1, 1]
        cx = intr[0, 2]
        cy = intr[1, 2]
        w = 2.0 * cx
        h = 2.0 * cy
        saved_fx = fx / w
        saved_fy = fy / h
        saved_cx = cx / w
        saved_cy = cy / h
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]

        w2c = world2cams[vid]
        camera.extend(w2c[:3].flatten().tolist())
        cameras.append(np.array(camera))

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }


if __name__ == "__main__":

    for stage in ("test",):
        intrinsics, world2cams, cam2worlds, near_fars = build_camera_info(
            list(range(10)), INPUT_IMAGE_DIR
        )

        keys = get_example_keys(stage)

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            print(
                f"Saving chunk {chunk_key} of {len(keys)} ({chunk_size / 1e6:.2f} MB)."
            )
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for key in keys:
            image_dir = INPUT_IMAGE_DIR / "Rectified" / key
            num_bytes = get_size(image_dir) // 7

            # Read metadata first to get timestamps
            example = load_metadata(intrinsics, world2cams)
            
            # Read images
            images = load_images(image_dir)
            
            # Merge the images into the example
            image_list = []
            for timestamp in example["timestamps"]:
                timestamp_item = timestamp.item()
                if timestamp_item in images:
                    image_list.append(images[timestamp_item])
                else:
                    # Insert a placeholder if image is missing
                    print(f"Warning: Missing image for timestamp {timestamp_item} in {key}")
                    image_list.append(torch.zeros(1, dtype=torch.uint8))  # Placeholder
            
            example["images"] = image_list
            
            # Convert timestamps tensor to a list for depth and normal loading
            timestamp_list = [t.item() for t in example["timestamps"]]
            
            # Read and add depth maps if available
            depths = load_depths(key, timestamp_list)
            
            if depths:
                depth_list = []
                for timestamp in example["timestamps"]:
                    timestamp_item = timestamp.item()
                    if timestamp_item in depths:
                        depth_list.append(depths[timestamp_item])
                    else:
                        # Insert a placeholder if depth is missing
                        print(f"Warning: Missing depth for timestamp {timestamp_item} in {key}")
                        depth_list.append(torch.zeros(1, dtype=torch.uint8))  # Placeholder
                
                example["depths"] = depth_list
                # If we have depths, update the size calculation
                if DEPTH_DIR and (DEPTH_DIR / key).exists():
                    depth_bytes = get_size(DEPTH_DIR / key)
                    num_bytes += depth_bytes
            else:
                example["depths"] = None
                
            # Read and add normal maps if available
            normals = load_normals(key, timestamp_list)
            
            if normals:
                normal_list = []
                for timestamp in example["timestamps"]:
                    timestamp_item = timestamp.item()
                    if timestamp_item in normals:
                        normal_list.append(normals[timestamp_item])
                    else:
                        # Insert a placeholder if normal map is missing
                        print(f"Warning: Missing normal map for timestamp {timestamp_item} in {key}")
                        normal_list.append(torch.zeros(1, dtype=torch.uint8))  # Placeholder
                
                example["normals"] = normal_list
                # If we have normals, update the size calculation
                if NORMAL_DIR and (NORMAL_DIR / key).exists():
                    normal_bytes = get_size(NORMAL_DIR / key)
                    num_bytes += normal_bytes
            else:
                example["normals"] = None

            # Add the key to the example
            example["key"] = key

            print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()

        # generate index
        print("Generate key:torch index...")
        index = {}
        stage_path = OUTPUT_DIR / stage
        for chunk_path in tqdm(list(stage_path.iterdir()), desc=f"Indexing {stage_path.name}"):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage_path))
        with (stage_path / "index.json").open("w") as f:
            json.dump(index, f)