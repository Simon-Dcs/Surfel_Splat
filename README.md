<p align="center">
  <h1 align="center">SurfelSplat: Learning Efficient and Generalizable  Gaussian Surfel Representations for Sparse-View Surface Reconstruction</h1>
  <h3 align="center">NeurIPS 2025</h3>
  <h3 align="center"><a href="https://arxiv.org/abs/2604.08370">Paper</a> | <a href="https://drive.google.com/file/d/11m6sbfPQDMKlIYYH2Z628UjGO1ASPuLm/view?usp=sharing">Pretrained Models</a></h3>
</p>


## Installation

Create a Python 3.10 environment and install the dependencies:

```bash
git clone https://github.com/Simon-Dcs/Surfel_Splat.git
cd Surfel_Splat
conda create -n surfelsplat python=3.10
conda activate surfelsplat
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install --no-build-isolation -r requirements_w_versions.txt
```

* If you encounter problems like `ModuleNotFoundError: No module named 'pkg_resources'`,try to run `pip install "setuptools<82"`
* If you encounter `numpy` version problems, please run `pip install numpy==1.26.3`

## Checkpoints

Download the SurfelSplat checkpoint [here](https://drive.google.com/file/d/11m6sbfPQDMKlIYYH2Z628UjGO1ASPuLm/view?usp=sharing) and place it under `checkpoints/`.

The UniMatch backbone weights are also required:

```bash
mkdir -p checkpoints
wget 'https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth' -P checkpoints
```

## Datasets

The codebase uses chunked dataset files in the same general style as pixelSplat. You should prepare your datasets into the format expected by the loaders in `src/dataset/`.

We present our preprocessed dataset [here](https://drive.google.com/file/d/1nA43vPn6yyVSyhyFkLdSEQlJlhdD_eJi/view?usp=sharing). By default, the config points to:

```text
datasets/torch_data
```

Update the dataset roots in the experiment config if your local dataset layout differs:

- `config/experiment/re10k.yaml`

## Inference

### Scene generation

The main entrypoint is:

- `generate.sh`


Available handles:

- `SCENE_ID`: integer scene id, for example `24`
- `CONTEXT_VIEWS`: comma-separated context view ids
- `TARGET_VIEWS`: comma-separated target view ids
- `CHECKPOINT_PATH`: checkpoint path
- `CUDA_DEVICE`: GPU id

Example:

```bash
SCENE_ID=24 \
CONTEXT_VIEWS=0,2,9 \
TARGET_VIEWS=4,2 \
CHECKPOINT_PATH=checkpoints/checkpoint.ckpt \
CUDA_DEVICE=0 \
bash generate.sh
```

You can also refer to dataset-specific script such as `generate_dtu.sh` and `generate_blendmvs.sh`.


## Training

The lightweight training wrapper is:

- `train.sh`

Example:

```bash
CUDA_DEVICES=0 \
BATCH_SIZE=1 \
bash train.sh
```

Before training, comment out the code at line 276 in `src/dataset/dataset_re10k.py` and line 529 in `src/model/encoder/encoder_costvolume.py`, since this release is configured to run with a global scale factor of `1/200`.
Adjust the experiment config, dataset roots, and batch size according to your hardware and dataset setup.

## Mesh Reconstruction

The mesh reconstruction utility is:

- `src/mesh/gs2mesh.py`

Example:

```bash
GS2MESH_INPUT_PLY=point_clouds/data/000000_scan24_train/gaussians.ply \
GS2MESH_OUTPUT_MESH=point_clouds/data/000000_scan24_train/scene24_mesh.ply \
GS2MESH_TEMP_MESH=point_clouds/data/000000_scan24_train/temp.ply \
GS2MESH_DOUBLE_SIDED=true \
GS2MESH_VISUALIZE=true \
python src/mesh/gs2mesh.py
```

## Evaluation

To evaluate a reconstructed DTU mesh, run:

```bash
python src/mesh/evaluate_single_scene.py \
  --input_mesh /path/to/mesh \
  --scan_id your_id \
  --output_dir /path/to/output \
  --DTU /path/to/DTU
```

The official DTU ground-truth data can be found [here](http://roboimagedata.compute.dtu.dk/?page_id=36).


## BibTeX

```bibtex
@inproceedings{daisurfelsplat,
  title={SurfelSplat: Learning Efficient and Generalizable Gaussian Surfel Representations for Sparse-View Surface Reconstruction},
  author={Dai, Chensheng and Zhang, Shengjun and Chen, Min and Duan, Yueqi},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
