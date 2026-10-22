# MoRe Full Attention 环境配置与复现指南（CUDA 12.8）

> **文档性质**：针对本地一次已成功完成多组推理的环境整理的可复现操作说明。**并非**官方完整训练环境，也**没有**验证 Stream/MagiAttention。
>
> **关键原则**：本地实际使用 **CUDA Toolkit 12.8.1 + PyTorch 2.9.0+cu128 + torchvision 0.24.0+cu128**；不要误写为 CUDA 11.8 或 12.9。
>
> 生成日期：2026-10-21。参见 `docs/fixes/2026-10-21-MoRe-inference-fix-log.md` 了解源码问题与补丁。

## 1. 已知硬件及路径

| 项目 | 实际/本次使用配置 |
| --- | --- |
| 宿主机 GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition，计算能力 `(12, 0)`，显存约 96 GB |
| 宿主机物理内存 | 240 GB（用户提供） |
| Docker 基础镜像 | `m.daocloud.io/docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` |
| 容器内存上限 | `128g`（Docker 以 GiB 量级计算；不是启动时预分配） |
| 容器 Swap 总上限 | `--memory-swap=128g`（与 `--memory` 相同时，不允许增加额外 Swap） |
| `/dev/shm` 上限 | 64 GiB；**实际共享内存消耗计入容器内存限制** |
| GPU | `device=0`，容器可访问该 GPU 的可用显存；Docker 参数并未预分配/限额为 96 GB |
| Python | Conda 环境 `more`，Python 3.10 |
| PyTorch / torchvision | `2.9.0+cu128` / `0.24.0+cu128` |
| PyTorch CUDA 编译版本 | `torch.version.cuda == '12.8'` |
| 宿主机挂载 | `/data/algorithm-research/MoRe` → `/workspace/MoRe` |
| 仓库根目录（容器） | `/workspace/MoRe/code/MoRe` |
| 数据与结果 | `/workspace/MoRe/data-tiny`、`/workspace/MoRe/results`（均在 Git 仓库外） |

**三个 CUDA 概念不要混淆**：`nvcc --version` 是容器 Toolkit 版本；`torch.version.cuda` 是 PyTorch wheel 的编译 CUDA 版本；`nvidia-smi` 显示的 CUDA Version 是驱动宣称的 CUDA 支持能力。三者不要求字面完全相同。宿主机的 NVIDIA **内核模块与用户态库版本**则必须正常匹配。

## 2. 宿主机：先检查 GPU 与 Docker

以下命令在**宿主机**执行，不是在容器中：

```bash
nvidia-smi
cat /proc/driver/nvidia/version
modinfo -F version nvidia
docker image inspect \
  m.daocloud.io/docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 \
  --format '{{.Id}} {{.Os}}/{{.Architecture}}'

docker run --rm --gpus '"device=0"' \
  m.daocloud.io/docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 \
  nvidia-smi
```

如果宿主机提示 `Failed to initialize NVML: Driver/library version mismatch`，**停止新建 GPU 容器**，优先诊断宿主机驱动。此次历史案例中，运行中的内核模块为 `595.84`，更新后的 NVML 和磁盘 DKMS 为 `595.99.02`；`nvidia-persistenced` socket 当时不存在。旧容器可以使用 GPU 并不代表新容器也会正常，因为旧容器可能仍挂载旧版用户态驱动库。**不要**在有 GPU 作业时盲目重启服务器、Docker 服务或卸载 NVIDIA 模块。可用维护窗口处理并复检。

## 3. 宿主机：创建容器（仅首次执行）

确认项目目录存在、没有 `more` 同名容器：

```bash
if [ ! -d /data/algorithm-research/MoRe/code/MoRe ]; then
  echo '错误：宿主机 MoRe 代码目录不存在，请先检查路径' >&2
elif docker container inspect more >/dev/null 2>&1; then
  echo 'more 容器已经存在，请直接 docker start more 或 docker exec -it more bash'
else
  docker run -itd \
    --name more \
    --hostname more \
    --gpus '"device=0"' \
    --memory=128g \
    --memory-swap=128g \
    --shm-size=64g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --mount type=bind,source=/data/algorithm-research/MoRe,target=/workspace/MoRe \
    --workdir /workspace/MoRe/code/MoRe \
    --restart unless-stopped \
    m.daocloud.io/docker.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 \
    sleep infinity
fi
```

启动或进入已有容器：

```bash
docker start more     # 仅当它尚未运行时执行；若显示已运行，直接跳过
# 宿主机执行：
docker exec -it more bash
```

注意：`docker run` 成功创建容器不代表 GPU 初始化或算法可用，必须完成后面的实测。`--restart unless-stopped` 会恢复容器进程，但**不会**自动恢复已结束的推理任务。

## 4. 容器内：系统依赖与 Conda

```bash
apt-get update
apt-get install -y --no-install-recommends \
  git ca-certificates curl wget build-essential cmake ninja-build pkg-config \
  ffmpeg libgl1 libglib2.0-0t64
```

- `git`：支持 `git+https://...` 的 LightGlue 安装。
- `libgl1`、`libglib2.0-0t64`：解决此次 `cv2` / `lightglue` 的运行时导入错误。
- CUDA Toolkit 与 `nvcc` 已由 `*-devel-*` 基础镜像提供，无需额外覆盖成 CUDA 11.8/12.9。

**Conda 的已知历史情况**：本次记录使用过 `/workspace/MoRe/others/Miniconda3-py312_26.7.1-1-Linux-x86_64.sh` 安装器，容器中 `more` 环境最终为 Python 3.10，运行路径为 `/root/miniconda3/envs/more/...`。安装器文件名中的 `py312` 指安装器的 base 解释器版本，**不等于** `more` 环境必须用 Python 3.12。

若新容器尚无 Conda，且上述安装器确实存在，可采用无交互方式（仅限目标目录尚不存在时）：

```bash
if [ ! -e /root/miniconda3 ] && \
   [ -f /workspace/MoRe/others/Miniconda3-py312_26.7.1-1-Linux-x86_64.sh ]; then
  bash /workspace/MoRe/others/Miniconda3-py312_26.7.1-1-Linux-x86_64.sh \
    -b -p /root/miniconda3
fi

source /root/miniconda3/etc/profile.d/conda.sh

# 首次创建环境时执行：
conda create -n more python=3.10 -y
conda activate more
python --version
which python
```

如果环境已存在，**不要**再次执行 `conda create -n more`，改为 `conda activate more`。如遇到 Conda 频道的 ToS 提示，按 Conda 显示的条款处理，不在脚本里跳过或伪造接受操作。

## 5. 容器内：安装 PyTorch 2.9.0 的 CUDA 12.8 构建

**本项目实际验证的是 CUDA 12.8，不是原版依赖列表的 CUDA 12.9。** 首次安装或重建环境时，在已激活的 `more` 环境中执行：

```bash
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  'torch==2.9.0' \
  'torchvision==0.24.0' \
  --index-url https://download.pytorch.org/whl/cu128
```

上传的历史配置文件还保留了 `torchaudio==2.9.0` 的安装命令，但本次对话没有提供它的最终版本验证。**Full Attention 这次复现不以 torchaudio 为必要条件**；确有音频依赖再安装对应 cu128 版本。

验证 CUDA 运算（不只是确认设备名）：

```bash
python - <<'PY'
import torch, torchvision
assert torch.__version__ == '2.9.0+cu128', torch.__version__
assert torchvision.__version__ == '0.24.0+cu128', torchvision.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch.cuda.is_available()
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
a = torch.randn(1024, 1024, device='cuda')
b = a @ a
torch.cuda.synchronize()
print('GPU 计算通过：', tuple(b.shape))
PY
```

**实际验证结果：** `torch==2.9.0+cu128`、`torchvision==0.24.0+cu128`、`torch.version.cuda==12.8`，RTX PRO 6000 Blackwell `(12,0)`，矩阵乘法成功。

## 6. 容器内：MoRe 适配依赖（重要）

项目维护以下三份文件，**作为此次复现环境的组成部分提交 Git**：

```text
requirements-cu128.txt
requirements-cu128-constraints.txt
requirements-git.txt
```

原版 `requirements.txt` 继续保留原样，方便对比上游。

### 6.1 依赖修改依据

- 原版锁定 `scipy==1.16.2`、`networkx==3.5`、`contourpy==1.3.3`，不满足本次 Python 3.10 环境；本地适配版本使用 `1.15.3`、`3.4.2`、`1.3.2`。
- 原版列出 CUDA 12.9 系列 `nvidia-*-cu12` 包；**不得**直接与现有 PyTorch 官方 cu128 wheel 的依赖混用或强制覆盖。
- PyTorch `2.9.0+cu128` 依赖的运行时库与 `triton` 由 PyTorch 安装和对应 constraints 管理。
- `nvidia-ml-py` 不是 `nvidia-*-cu12` CUDA runtime 包，不能以同样规则批量删除。
- Git 版 LightGlue 应固定此次已安装成功的提交 `2f23ca2ea9638cecad7f7220795210fc6b8353c3`，而非浮动分支。

**从零安装的必要前提**：先把三份适配文件从本次 Git 提交取回。仅有原版 `requirements.txt` 时，不能直接跳到本节的 `-r requirements-cu128.txt` 命令；也不能假设新环境中的约束文件可自动生成出与旧环境完全相同的版本组合。

### 6.2 复核文件并安装

```bash
cd /workspace/MoRe/code/MoRe
conda activate more

python - <<'PY_CHECK'
from pathlib import Path
names = ('requirements-cu128.txt', 'requirements-cu128-constraints.txt', 'requirements-git.txt')
missing = [name for name in names if not Path(name).is_file() or Path(name).stat().st_size == 0]
if missing:
    raise SystemExit('适配依赖文件不存在或为空：' + ', '.join(missing))
print('依赖文件检查通过')
PY_CHECK

# 检查 Git 依赖是否已经锁定到准确提交
cat requirements-git.txt

# 先模拟解析；确认没有需要替换 torch/cu128 的冲突
python -m pip install --dry-run \
  -r requirements-cu128.txt \
  -c requirements-cu128-constraints.txt \
  --default-timeout=1000

# 模拟成功后再执行真正安装
python -m pip install \
  -r requirements-cu128.txt \
  -c requirements-cu128-constraints.txt \
  --default-timeout=1000

# LightGlue 的第三方依赖应由上面的普通依赖清单先满足；
# --no-deps 防止这一步意外覆盖已验证的 PyTorch
python -m pip install --no-deps -r requirements-git.txt
python -m pip check
```

建议使用单个经确认的 PyPI 镜像或官方 PyPI 索引，不把多个 `--extra-index-url` 当作确定的下载优先级；`--index-url` 的选择属于网络环境配置，不改变 CUDA wheel 的准确来源。

### 6.3 约束快照的生成方式（用于检查/更新，不要盲目覆盖 Git 版本）

```bash
python -m pip freeze \
  | grep -Ei '^(torch|torchvision|triton|nvidia-.*-cu12)==' \
  > /tmp/more-cu128-constraints.current.txt

diff -u requirements-cu128-constraints.txt \
  /tmp/more-cu128-constraints.current.txt || true
```

此文件是当前平台和本次 wheel 组合的版本约束，不是独立安装所有 CUDA 包的命令。若需要更新，先完成 CUDA 计算回归测试和 `pip check`，然后再人工审查差异。

## 7. Full Attention 代码适配与验证

必须包含本地三个源码补丁（具体原因和 diff 见问题记录）：

```text
vggt/layers/attention.py      # 将 MagiAttention 改为 SparseAttention 内按需导入
vggt/models/aggregator.py    # 去掉无效 save_attn 关键字参数
inference.py                 # 正确处理 NumPy 点云，并跳过未实现的空 GLB 导出
```

```bash
python -m py_compile \
  vggt/layers/attention.py \
  vggt/models/aggregator.py \
  inference.py

python -m pip check
python - <<'PY'
import cv2, lightglue, kornia, numpy, scipy, trimesh, hydra, onnxruntime
from vggt.models.vggt import VGGT
print('核心依赖与 VGGT 导入成功')
PY

python - <<'PY'
from hydra import initialize, compose
with initialize(version_base=None, config_path='training/config'):
    cfg = compose(config_name='omniworld_full')
assert cfg.model.block_size == 0
print('Full Attention 配置校验通过，block_size=', cfg.model.block_size)
PY
```

`_self_` 的 Hydra UserWarning 与 AMP 的 FutureWarning 在上述实际测试中属于非阻断警告，**不应**因为它们存在就盲目更改配置合并规则或 PyTorch 版本。

## 8. 权重、输入和完整推理

这两个路径应存在，但**权重不加入 Git**：

```text
pretrained/more_full.pt
pretrained/skyseg.onnx
```

真实模型权重的来源应遵循上游项目发布页，首次使用请校验下载文件。建议在本地留存 SHA256 而非把二进制模型加入代码仓库：

```bash
sha256sum pretrained/more_full.pt pretrained/skyseg.onnx
```

输入是排序后的 `.jpg/.jpeg/.png` 图片序列，不是直接传入 MP4 文件。示例：

```bash
set -o pipefail
python inference.py \
  --config_path training/config/omniworld_full.yaml \
  --ckpt_path pretrained/more_full.pt \
  --image_path /workspace/MoRe/data-tiny/test01 \
  --output_dir /workspace/MoRe/results/test01_full_final \
  --skyseg_onnx pretrained/skyseg.onnx \
  --conf_thres 50.0 \
  --predict_motion \
  2>&1 | tee /workspace/MoRe/results/test01_full_final.log
```

**必须使用尚无旧结果的输出目录**，以避免把以前失败运行的文件算入新结果。判断成功不能只看 `Inference completed`，还要有 `Processing finished`，并检查每帧 PLY、depth、extrinsic、intrinsic、motion mask 的实际内容。

GLB：此次修复**跳过**无几何体的 `scene.glb`；没有 GLB 不代表本次 Full Attention 推理失败。需要真实 GLB/融合场景应另开任务实现和验证。

## 9. 环境快照、版本检查与代码管理

可将完整环境快照存放在**仓库外**的挂载目录，避免无意提交带有本机路径、私有索引或凭据的日志：

```bash
mkdir -p /workspace/MoRe/others/more-env-snapshots

python -m pip freeze \
  > /workspace/MoRe/others/more-env-snapshots/pip-freeze.txt

conda env export --no-builds \
  > /workspace/MoRe/others/more-env-snapshots/conda-env-full.yml

nvidia-smi \
  > /workspace/MoRe/others/more-env-snapshots/nvidia-smi.txt

cd /workspace/MoRe/code/MoRe
git rev-parse HEAD
git status --short
```

Conda 的完整环境导出常包含当前环境的绝对 `prefix`；若未来将快照提交到仓库，先**删除或泛化**这类字段，并检查 URL 凭据，不要直接提交未经审阅的原始日志。

**推荐进入 Git 的文件**：三处源码补丁、上述三份 requirements 文件、本指南、纠错记录、经审阅的 `.gitignore`。原版 `requirements.txt` 不动。

**推荐排除**：权重 `pretrained/`、结果、`__pycache__/`、本次备份 `*.bak-*`、一次性修复脚本，以及父目录 `/workspace/MoRe/others/` 下的下载安装包与原始环境快照。

参阅 `docs/fixes/2026-10-21-MoRe-inference-fix-log.md` 中的回归检查和提交约束。执行 Git 提交前要分别审查**工作区差异**和**暂存区差异**，避免把 `*.pt`、`*.onnx`、`*.npy` 等大文件或包含凭据的快照带进提交。

## 10. 文档来源与已知限制

- 用户上传的历史《MoRe 环境配置命令》，其中记载 Docker 参数、Miniconda 安装器、cu128 PyTorch 安装、三个适配依赖文件和早期验证结果。
- 本次 2026-10-20—21 终端命令和异常记录。
- [上游 requirements.txt](https://github.com/HellexF/MoRe/blob/main/requirements.txt)、[inference.py](https://github.com/HellexF/MoRe/blob/main/inference.py)、[aggregator.py](https://github.com/HellexF/MoRe/blob/main/vggt/models/aggregator.py)。

**尚未验证**：Stream/MagiAttention 编译与执行、重新训练、跨容器全新离线重建、动态几何精度、全局 GLB 导出。此次“推理成功”依据用户报告；精确的帧数/耗时/质量请以保存的真实日志与生成文件为准。
