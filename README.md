# PyTorch 遥感分割最小训练与验证示例

GitHub 仓库：https://github.com/liurijin123/pytorch-remote-sensing-segmentation-pipeline

仓库状态：已于 2026-08-15 上传并核对远端 `main` 分支。

配套文章：《PyTorch 遥感分割训练流程：为什么需要 Dataset、DataLoader 与验证》

本示例承接上一篇“VS Code + pip GPU 深度学习环境验证”，适用于 Windows 10/11 64 位、NVIDIA GPU 和 Python 3.13。

代码只演示一条最小流程：读取 256×256 三波段 GeoTIFF、组织训练与验证批次、更新模型参数、按验证损失保存最佳权重，再对一张测试影像输出预测 PNG。合成数据不能代表真实遥感任务的模型精度。

## 文件与数据

```text
make_demo_data.py  # 生成训练、验证和测试数据
dataset.py         # 定义一条影像与标签样本
model.py           # 保持空间尺寸不变的三层卷积网络
main.py            # DataLoader、训练、验证、最佳权重和预测

data/
├─ train/images、train/labels   # 12 对训练样本
├─ val/images、val/labels       # 4 对验证样本
└─ test/images                  # 1 张无标签测试影像
```

训练集和验证集复用同一个 `RemoteSensingDataset`。训练 DataLoader 使用 `shuffle=True`，验证 DataLoader 使用 `shuffle=False`。验证阶段不执行反向传播和参数更新，程序保存 `val_loss` 最低时的参数。

## 在 VS Code 中运行

用 VS Code 打开本目录，在集成终端依次执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --site set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --site list
python -m pip install -r requirements-torch-cu126.txt
python -m pip install -r requirements-tools.txt
python -m pip check
python make_demo_data.py
python main.py
```

镜像设置只写入当前 `.venv`。`requirements-torch-cu126.txt` 指定 PyTorch 官方 CUDA 12.6 索引，因此安装 PyTorch 时不会改用普通 PyPI 镜像。取消当前环境的镜像配置可执行：

```powershell
python -m pip config --site unset global.index-url
```

如果 PowerShell 禁止执行激活脚本，可将 VS Code 终端切换为“命令提示符”，再执行：

```bat
.venv\Scripts\activate.bat
```

成功运行后生成：

```text
data/                           # 自动生成的演示数据
outputs/best_tiny_segnet.pth    # 验证损失最低的模型权重
outputs/prediction.png          # 测试影像的黑白预测图
```

终端应显示批次形状、每轮 `train_loss` 与 `val_loss`、最佳权重路径和预测图路径。代码强制使用 `cuda:0`；没有可用 NVIDIA GPU 时会停止，不自动回退到 CPU。

## 验证状态

已于 2026-08-15 完成端到端运行，实际环境如下：

- Python 3.13.1；
- PyTorch 2.12.1+cu126；
- torchvision 0.27.1+cu126；
- Rasterio 1.5.1；
- NumPy 2.4.4；
- Pillow 12.2.0；
- NVIDIA GeForce RTX 2070，PyTorch CUDA runtime 12.6；
- `torch.cuda.is_available()` 为 `True`，CUDA 张量位于 `cuda:0`。

实际生成 12 对训练影像与标签、4 对验证影像与标签和 1 张测试影像。单个批次影像形状为 `(4, 3, 256, 256)`，标签形状为 `(4, 256, 256)`。

30 轮训练中，第 1 轮 `train_loss/val_loss` 为 `0.7392/0.7191`，第 30 轮为 `0.0117/0.0107`。`outputs/best_tiny_segnet.pth` 已成功保存和重载；`prediction.png` 为 256×256，像元值包含 `0` 和 `255`。

上述数值只证明合成数据上的代码闭环和基本学习行为，不是 IoU 或真实遥感精度。示例版本核对于 2026-08-15；若 PyTorch 官方安装命令变化，应更新依赖文件并重新验证。

pip 镜像配置参考：清华大学开源软件镜像站 PyPI 使用帮助（https://mirrors.tuna.tsinghua.edu.cn/help/pypi/）。
