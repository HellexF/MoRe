<p align="center" />
<h1 align="center">MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer</h1>

<p align="center">
    <a href=""><strong>Juntong Fang*</strong></a>
    ·
    <a href=""><strong>Zequn Chen*</strong></a>
    ·
    <a href="https://weiqi-zhang.github.io"><strong>Weiqi Zhang*</strong></a>
    ·
    <a href=""><strong>Donglin Di</strong></a>
    ·
    <a href=""><strong>Xuancheng Zhang</strong></a>
    ·
    <a href=""><strong>Chengmin Yang</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu†</strong></a>
</p>
<h2 align="center">CVPR 2026</h2>
<h3 align="center"><a href="https://arxiv.org/abs/2603.05078">Paper</a> | <a href="https://hellexf.github.io/MoRe/">Project Page</a></h3>
<div align="center"></div>
<div align="center"></div>
<p align="center">
    <img src="media/cover.pdf" width="780" />
</p>

## 🚀 News
* **[2026.03]** Refactored inference scripts and pretrained weights are released.
* **[2026.02]** MoRe has been accepted by **CVPR 2026**!

---

## 📖 Introduction

MoRe is a feedforward 4D reconstruction transformer designed to efficiently recover dynamic 3D scenes from monocular videos. 

- **Motion-Structure Disentanglement**: Employs an attention-forcing strategy to separate dynamic motion from static structure.
- **Grouped Causal Attention**: Captures temporal dependencies and adapts to varying token lengths for coherent geometry.

<p align="center">
    <img src="media/visualization.pdf" width="800" />
</p>

---

## 🛠️ Setup

### Installation

Clone the repository and create an anaconda environment using
```shell
# Clone the repository
git clone https://github.com/HellexF/MoRe
cd MoRe

# Create and activate environment
conda create -n more python=3.10 -y
conda activate more

# Install PyTorch and CUDA toolkit
conda install pytorch=2.9.0 torchvision=0.24.0 cudatoolkit=11.8 -c pytorch
conda install cudatoolkit-dev=11.8 -c conda-forge

# Install remaining dependencies
pip install -r requirements.txt
```

**Required Extension** : We use [MagiAttention](https://github.com/SandAI-org/MagiAttention) for implementing grouped causal attention. Please follow their installation guide to enable stream inference.

### Pretrained Model
We provide the pretrained full attention and stream models. Please download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1T5CBo4bqJAaR0IBU-v8fde87FkqdDpUe) and place them in the `./pretrained` directory:

## 💻Inference

```shell
python inference.py \
    --config_path training/config/omniworld_full.yaml \
    --ckpt_path pretrained/more_full.pt \
    --image_path ./data/example_video \
    --output_dir ./results/full_res \
    --conf_thres 50.0 \
    --predict_motion
```

## 🏋️Training
We offer the training config for both full attention and stream training on Omniworld-Game dataset. Please refer to the [Omniworld](https://github.com/yangzhou24/OmniWorld) for downloading and place it in the './dataset' directory. To train full attention version, simply run

```shell
torchrun --nproc_per_node=$GPU_NUM training/launch.py --config omniworld_full
```

Similarly, to train the stream versiob, run 
```shell
torchrun --nproc_per_node=$GPU_NUM training/launch.py --config omniworld_stream
```

## 📊Evaluation

Run the following scripts to evaluate benchmarks for camera poses and video depth:
```shell
# Camera Pose Evaluation
bash eval/relpose/run.sh

# Video Depth Evaluation
bash eval/video_depth/run.sh
```

## 📑Acknowledgements

This project is built upon [VGGT](https://github.com/facebookresearch/vggt), [MagiAttention](https://github.com/SandAI-org/MagiAttention).  We thank all the authors for their great repos.


## ✒️Citation

If you find our code or paper useful, please consider citing
```bibtex
@misc{fang2026moremotionawarefeedforward4d,
      title={MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer}, 
      author={Juntong Fang and Zequn Chen and Weiqi Zhang and Donglin Di and Xuancheng Zhang and Chengmin Yang and Yu-Shen Liu},
      year={2026},
      eprint={2603.05078},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.05078}, 
}
```

