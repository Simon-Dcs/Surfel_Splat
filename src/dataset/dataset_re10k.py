import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8,Int
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    test_times_per_scene: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 3000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.single_scene is not None:
            chunk_path = self.index[self.cfg.single_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            self.chunks = self.chunks[:: cfg.test_chunk_interval]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # print(chunk_path)
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.single_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.single_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            # for example in chunk:
            times_per_scene = self.cfg.test_times_per_scene
            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                if times_per_scene > 1:  # specifically for DTU
                    scene = f"{example['key']}_{(run_idx % times_per_scene):02d}"
                else:
                    scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    # reverse the context
                    # context_indices = torch.flip(context_indices, dims=[0])
                    # print(context_indices)
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                #print('IMPORTANT',context_images.shape)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)

                # DEPTH
                context_depth = [
                    example["depths"][index.item()] for index in context_indices
                ]
                context_depth = self.convert_depth(context_depth)
                #print('IMPORTANT',context_depth.max())
                target_depth = [
                    example["depths"][index.item()] for index in target_indices
                ]
                target_depth = self.convert_depth(target_depth)

                # Normal
                context_normal = [
                    example["normals"][index.item()] for index in context_indices
                ]
                context_normal = self.convert_images(context_normal)
                # print('IMPORTANT',context_normal.shape)
                # print(context_normal.mean())
                target_normal = [
                    example["normals"][index.item()] for index in target_indices
                ]
                target_normal = self.convert_images(target_normal)


                #Load the mask
                # context_mask = [
                #     example["masks"][index.item()] for index in context_indices
                # ]
                # context_mask = self.convert_depth(context_mask)
                # target_mask = [
                #     example["masks"][index.item()] for index in target_indices
                # ]
                # target_mask = self.convert_depth(target_mask)     



                # print('depth is ',target_depth.max())UInt8
                # print('depth shape is ',target_depth.shape)
                # Skip the example if the images don't have the right shape.

                shape_list = [(3, 480, 640),(3, 360, 640),(3,270,480),(3, 512, 640),(3, 576, 768)]
                context_image_invalid = context_images.shape[1:] not in shape_list
                target_image_invalid = target_images.shape[1:] not in shape_list



                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
                    a, b = context_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_epsilon:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                        continue
                    #print('sts',scale)
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                        "depth": context_depth,
                        "normal": context_normal,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                        "depth": target_depth,
                        "normal": target_normal,
                    },
                    "scene": scene,
                }
                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx 
        intrinsics[:, 1, 1] = fy 
        intrinsics[:, 0, 2] = cx 
        intrinsics[:, 1, 2] = cy 

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        
        w2c[:,:3,-1] *= 200      # Remember to comment out this line when training

        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)


    def convert_mask(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 1 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def convert_depth(
        self,
        images: list[UInt8[Tensor, "..."]],
    ):# -> Float[Tensor, "batch 1 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.single_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.single_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}
                #print(index.keys())
                #print(merged_index.keys())
                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()) *
                self.cfg.test_times_per_scene, self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.test_times_per_scene
        )
