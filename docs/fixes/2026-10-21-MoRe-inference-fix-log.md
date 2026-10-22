# MoRe Full Attention 复现问题与修复记录

- **记录时间**：2026-10-20—2026-10-21
- **项目**：[HellexF/MoRe](https://github.com/HellexF/MoRe)
- **验证范围**：Docker 中的单卡 Full Attention 推理；**不包含** Stream/MagiAttention 的构建与功能验证、训练复现或定量精度评估。
- **运行环境摘要**：NVIDIA RTX PRO 6000 Blackwell Workstation Edition（Compute Capability 12.0，约 96 GB 显存）；Ubuntu 24.04 / CUDA Toolkit 12.8.1 / Python 3.10 / PyTorch `2.9.0+cu128` / torchvision `0.24.0+cu128`。
- **事实边界**：FIX-001—FIX-003 的错误与修复过程有实际命令输出；用户在 2026-10-21 反馈随后多组推理已成功，但尚未在本记录中附上这些成功运行的日志、文件数审计和质量指标，故不声称完成定量评测。

## 1. 错误与解决结果总览

| 编号 | 阶段 | 症状 | 根因 | 处理 | 状态 |
| --- | --- | --- | --- | --- | --- |
| ENV-001 | 宿主机 GPU / Docker | 宿主机 `nvidia-smi` 报 `Driver/library version mismatch`；新容器创建时找不到 `nvidia-persistenced/socket` | 内核实际加载 `595.84`，宿主机 NVML 与磁盘 DKMS 为 `595.99.02`；Persistence Daemon 当时未运行 | 先排查版本一致性与 daemon 状态，维护窗口内使内核模块及用户态库版本一致；不要用改容器 CUDA 版本来掩盖宿主机驱动问题 | 根因有日志证据；具体宿主机修复操作未提供，不作已执行声明 |
| ENV-002 | 基础工具 | pip 安装 Git 版 LightGlue 时 `Cannot find command 'git'` | 精简 CUDA 镜像无 Git 可执行文件 | 安装 `git` 与 CA 证书，重新安装 Git 依赖 | 已解决 |
| ENV-003 | 依赖管理 | 原版 requirements 与 Python 3.10 和现有 cu128 环境不一致 | 原版固定 Python 3.11+ 的部分依赖，并额外固定 CUDA 12.9 系列运行时 | 使用 `requirements-cu128.txt`、约束文件和独立 Git 依赖；保留官方 requirements | 已部署于本次环境；完整复现以仓库中的实际文件为准 |
| ENV-004 | OpenCV / LightGlue | `libGL.so.1: cannot open shared object file` | 容器缺少系统共享库 | `apt-get install libgl1 libglib2.0-0t64` | `cv2`、`lightglue`、`kornia` 导入通过 |
| FIX-000 | 模型导入 | `ModuleNotFoundError`（`magi_attention`） | `attention.py` 模块顶层强制加载仅 Sparse/Stream 路径使用的扩展 | 将 `flex_flash_attn_func` 导入移入 `SparseAttention.forward()` | VGGT 导入、Full 配置解析通过；Stream 未验证 |
| FIX-001 | 模型前向 | `_process_global_attention() got an unexpected keyword argument 'save_attn'` | 调用处多传参数，目标方法未声明 | 仅去掉 Global Attention 调用处的 `save_attn=` | 已通过真实推理越过原报错点 |
| FIX-002 | 结果导出 | `'numpy.ndarray' object has no attribute 'cpu'` | 几何反投影函数已返回 `numpy.ndarray`，导出再次调用 Tensor 方法 | 将 `pts` 转为 `np.asarray(..., dtype=np.float32)` | 已通过真实推理越过原报错点 |
| FIX-003 | 结果导出 | `Can't export empty scenes!` | `trimesh.Scene()` 中没有几何体却直接导出 GLB | 跳过尚未实现的空场景 GLB 导出，继续保留逐帧输出 | 用户反馈之后多组程序成功；未附可复核文件审计 |

## 2. 环境依赖相关问题

### ENV-001：宿主机驱动模块与 NVML 用户态库不一致

**观测证据：**

```text
宿主机 nvidia-smi：Failed to initialize NVML: Driver/library version mismatch
宿主机 NVML：595.99（内核日志进一步确认 API client 595.99.02）
/proc/driver/nvidia/version：595.84
modinfo -F version nvidia：595.99.02
DKMS：595.99.02，已为 6.8.0-138/147-generic 安装
旧 colmap-gpu 容器：nvidia-smi 595.84，可与运行中的旧内核模块通信
新 more 容器：/run/nvidia-persistenced/socket 不存在，OCI 挂载失败
```

**分析：** Docker 容器共享宿主机内核，但有自己的文件/挂载视图。旧容器可能仍持有升级前挂载的旧版用户态库，故旧容器 `nvidia-smi` 可以成功，并不表示宿主机驱动栈一致。`nvidia-persistenced` 未运行可解释当时 socket 缺失，但不能据此断言 daemon 停止完全由版本不匹配引起。

**安全处理：** 检查 `cat /proc/driver/nvidia/version`、`modinfo -F version nvidia`、`dkms status`、`journalctl -k -b`；确认旧容器中无正在运行的任务，并安排维护窗口后再重启宿主机/恢复 daemon，重启后重新验证驱动版本与 GPU 透传。**不要**在有活跃 GPU 作业的服务器上直接卸载驱动模块或执行 `apt purge 'nvidia*'`。未获取到最终驱动修复命令，不将任何特定重启操作记为“已执行”。

### ENV-002：Git 依赖缺少 Git 命令

```text
Collecting lightglue
Cloning https://github.com/jytime/LightGlue.git
ERROR: Cannot find command 'git'
```

修复：

```bash
apt-get update
apt-get install -y --no-install-recommends git ca-certificates
```

之后 Git 版本 LightGlue 安装成功，记录的提交为 `2f23ca2ea9638cecad7f7220795210fc6b8353c3`。应在 `requirements-git.txt` 中锁定该提交；仅使用默认分支 URL 不能保证下次安装同一代码。

### ENV-003：原版 requirements 不适合直接用于当前环境

原版 `requirements.txt` 包含 `scipy==1.16.2`、`networkx==3.5`、`contourpy==1.3.3` 等不适用于 Python 3.10 的固定版本，同时固定了 `nvidia-cuda-runtime-cu12==12.9.79`、`nvidia-cublas-cu12==12.9.1.4` 等 CUDA 12.9 系列用户态依赖。这些固定包与已装的 PyTorch `2.9.0+cu128` 组合不应未经解析就直接混装。

处理方式：

1. **原版 `requirements.txt` 保持不变**，便于对照上游。
2. 项目专用 `requirements-cu128.txt` 将已验证的 Python 3.10 兼容版本设为 `scipy==1.15.3`、`networkx==3.4.2`、`contourpy==1.3.2`，并将部分非核心推理包和固定 CUDA 12.9 依赖从本次安装列表中拆出。
3. 使用 `requirements-cu128-constraints.txt` 约束现有 `torch`、`torchvision`、`triton` 和 `nvidia-*-cu12` 版本；先安装官方 `cu128` PyTorch wheel，再安装普通依赖。
4. 将 LightGlue 的 Git 依赖独立安装，并运行 `python -m pip check`、实际导入与 CUDA 计算测试。

**说明：** `pip check` 通过只表示已安装 Python 包声明的版本关系一致，不验证 Linux 动态库、CUDA 扩展 ABI 或算法输出质量。

### ENV-004：OpenCV / LightGlue 缺少 libGL

```text
[FAIL] cv2       ImportError: libGL.so.1: cannot open shared object file
[FAIL] lightglue ImportError: libGL.so.1: cannot open shared object file
```

修复：

```bash
apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64
ldconfig -p | grep 'libGL.so.1'
python -c 'import cv2, lightglue, kornia; print("OpenCV / LightGlue / Kornia OK")'
```

实测版本：OpenCV `4.11.0`、Kornia `0.8.2`；LightGlue 导入成功。`torch.cuda.amp.custom_fwd` 的 `FutureWarning` 不影响本次导入，但后续可单独更新 API。

## 3. 源码级修复（提交到 Git 的内容）

### FIX-000：MagiAttention 改为按需导入

**文件：** `vggt/layers/attention.py`。

原源码在文件顶层导入：

```python
from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
from magi_attention.common.enum import AttnMaskType
```

Full Attention 使用普通 `Attention` 路径，并不在该路径执行 `flex_flash_attn_func`；但模块顶层导入使得即使只想运行 Full Attention 也必须首先安装 MagiAttention。调整方法：移除上述顶层两处导入，将 `flex_flash_attn_func` 导入放入 `SparseAttention.forward()` 方法开头；原来的 `AttnMaskType` 在本次检查的源码中未被实际引用，因此无需再导入。

```python
class SparseAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, motion_scores=None,
                save_attn=False, kv_cache=None) -> Tensor:
        from magi_attention.functional.flex_flash_attn import flex_flash_attn_func
        # 后续原算法代码不变
```

**实际验证：** VGGT 模块导入成功；Hydra `omniworld_full` 的 `cfg.model.block_size == 0`；Full Attention 前向越过该导入。**不表示** Stream / SparseAttention 功能已通过验证。

### FIX-001：Global Attention 函数接口不匹配

**文件：** `vggt/models/aggregator.py`。

原始调用处：

```python
self._process_global_attention(
    tokens, B, S, P, C, global_idx,
    pos=pos,
    save_intermediate=id in save_ids,
    save_attn=P if id == 23 else 0,
)
```

被调用函数签名为：

```python
def _process_global_attention(self, tokens, B, S, P, C,
                              global_idx, pos=None, save_intermediate=False):
```

其签名不接受 `save_attn`，函数体内也未读取该参数。删除**仅 Global Attention 调用**中的 `save_attn=P if id == 23 else 0`：

```diff
- tokens, B, S, P, C, global_idx, pos=pos, save_intermediate=id in save_ids, save_attn=P if id == 23 else 0
+ tokens, B, S, P, C, global_idx, pos=pos, save_intermediate=id in save_ids
```

不要删除 `_process_frame_attention()` 的同名参数，它在该路径中是合法的。

**验证：** `python -m py_compile vggt/models/aggregator.py` 通过；同样的 9 帧推理进入 Export 阶段，不再出现该 `TypeError`。只修改传参，不改变 Global Attention 内部计算。

### FIX-002：点云导出使用错误的数据类型 API

**文件：** `inference.py`。

`vggt/utils/geometry.py` 的 `unproject_depth_map_to_point_map()` 已在内部把 Tensor 转为 NumPy，并使用 `np.stack()` 返回 `numpy.ndarray`。导出时代码却继续调用 Tensor 的 `.cpu().float().numpy()`：

```diff
- pts = point_map_unproj[i].reshape(-1, 3).cpu().float().numpy()
+ pts = np.asarray(point_map_unproj[i], dtype=np.float32).reshape(-1, 3)
```

**验证：** 修改后 9 帧推理越过旧异常，继续执行到最后的 GLB 导出。类型问题修复有实测证据；真实点云的空间精度尚未独立核验。

### FIX-003：不应导出没有几何体的 GLB 场景

**文件：** `inference.py`。

原始末尾代码：

```python
scene = trimesh.Scene()
# Integration of alignment and camera visualization could go here
scene.export(os.path.join(self.args.output_dir, 'glb', 'scene.glb'))
```

未添加 `PointCloud` / `Trimesh` 等任何 geometry，因而 `trimesh` 抛出 `Can't export empty scenes!`。本次采用**最小修复**：跳过尚未实现的全局 GLB 合成，保留逐帧 PLY、深度、相机内外参及运动掩码导出。

```diff
- # Optional: Save Global Scene (GLB)
- scene = trimesh.Scene()
- # Integration of alignment and camera visualization could go here
- scene.export(os.path.join(self.args.output_dir, 'glb', 'scene.glb'))
+ # The original GLB export constructed an empty scene and always failed.
+ # Per-frame PLY files are the actual point-cloud outputs; do not write a fake GLB.
+ print("[*] Skipping global GLB: scene aggregation is not implemented.")
```

**验证边界：** 用户反馈后续多组程序已成功；本记录没有对应成功日志或输出审计结果。此修复不会生成 `scene.glb`，也**不表示**已实现全局统一场景融合或动态点云动画。若将来需要 GLB，应把真实顶点和颜色添加到场景，并验证坐标系和几何范围，而非导出空占位文件。

## 4. 已运行的验证命令与判据

```bash
cd /workspace/MoRe/code/MoRe

python -m py_compile \
  inference.py \
  vggt/layers/attention.py \
  vggt/models/aggregator.py

python -m pip check
python -c 'from vggt.models.vggt import VGGT; print("VGGT import OK")'

git diff --check
```

对成功的 Full Attention 推理，使用**全新结果目录**，防止混入前一次失败的部分输出：

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

**注意：** 原版 `inference.py` 在主程序 `except` 中仅打印 `[Error]`，可能仍返回退出码 0。除检查 shell 退出码外，还必须检查日志中确实出现 `Processing finished`，并审核每帧文件存在性、PLY 非空、深度数值有效、相机矩阵维度和运动掩码可读性。`Inference completed in 0.45s` 不等同于端到端耗时（GPU 异步计算与导出不在完整计时范围内）。

## 5. 非阻断警告与后续独立工作

- Hydra 警告 `Defaults list is missing _self_`：已能解析 Full Attention 配置；修改默认合并顺序应单独验证，不在这次补丁中随意更改。
- `torch.cuda.amp.autocast` / LightGlue `torch.cuda.amp.custom_fwd` 的 FutureWarning：目前不影响测试；后续可单独迁移至 `torch.amp.*`，并验证数值一致性。
- `model.load_state_dict(..., strict=False)`：可能掩盖权重键缺失/多余，建议后续增加 `missing_keys`、`unexpected_keys` 检查和记录。
- Stream Attention / MagiAttention：本次未配置或验证；不可用 Full Attention 的成功证明 Stream 路径正常。
- 全局 GLB：目前主动跳过空 GLB；如要输出可见 GLB，需要另行实现真实几何体聚合。

## 6. 提交范围与复现约束

应提交：

```text
inference.py
vggt/layers/attention.py
vggt/models/aggregator.py
docs/fixes/2026-10-21-MoRe-inference-fix-log.md
docs/setup/MoRe-CUDA128-environment.md
requirements-cu128.txt
requirements-cu128-constraints.txt
requirements-git.txt
.gitignore（仅经人工检查后的项目级忽略规则）
```

不要提交：`pretrained/` 中的权重、`*.bak-*` 备份、`__pycache__/`、测试数据、推理结果以及一次性本地修复脚本。`requirements-cu128-constraints.txt` 属于此环境的 CUDA 依赖约束快照，应在提交前检查没有 `file://`、私有 URL、本机绝对路径或凭据。

在提交前审查：`git diff --cached --check`、`git diff --cached --stat`、`git diff --cached` 与 `git status --short`。**不得用 `git add .` 批量提交未知文件，也不应为清理而运行 `git clean -fdx`。**

## 7. 来源与限制

- [MoRe 官方仓库](https://github.com/HellexF/MoRe)
- [官方 inference.py](https://github.com/HellexF/MoRe/blob/main/inference.py)
- [官方 aggregator.py](https://github.com/HellexF/MoRe/blob/main/vggt/models/aggregator.py)
- [官方 geometry.py](https://github.com/HellexF/MoRe/blob/main/vggt/utils/geometry.py)
- 本次配置/运行过程的终端日志以及上传的《MoRe 环境配置命令》。

本记录说明本地分支相对上游的修复理由；**不是**上游项目已经合并这些补丁的证明。完整复现应同时记录 `git rev-parse HEAD`、预训练权重 SHA256、实际运行参数、输入数据清单与环境快照。
