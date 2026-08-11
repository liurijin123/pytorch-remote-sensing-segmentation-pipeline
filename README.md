# PyTorch 遥感分割最小示例

GitHub 仓库：<https://github.com/liurijin123/pytorch-remote-sensing-segmentation-pipeline>

仓库状态：已于 2026-08-11 上传并验证 `main` 分支。

配套文章：《PyTorch 遥感分割代码结构：Dataset、DataLoader 与训练流程》

本示例承接上一篇“VS Code + pip GPU 深度学习环境验证”，当前实际运行环境为 Windows 10/11 64 位、NVIDIA GPU、Python 3.13。

这套代码只演示最基本的数据流：读取 256×256 三波段 GeoTIFF、组织批次、在 NVIDIA GPU 上训练、保存权重并输出一张预测 PNG。

## 文件

```text
make_demo_data.py  # 生成 256×256 三波段影像和二分类标签
dataset.py         # 自定义 Dataset
model.py           # 三层卷积分割网络
main.py            # DataLoader、训练、保存、加载和预测
```

## 在 VS Code 中运行

用 VS Code 直接打开本目录，然后在集成终端依次执行：

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

上述镜像设置只写入当前 `.venv`。`requirements-torch-cu126.txt` 内部指定了 PyTorch 官方 CUDA 12.6 索引，因此安装 PyTorch 时不会改用普通 PyPI 镜像。若要取消当前环境的镜像配置，执行：

```powershell
python -m pip config --site unset global.index-url
```

如果 PowerShell 禁止执行激活脚本，可以将 VS Code 终端切换为“命令提示符”，再执行：

```bat
.venv\Scripts\activate.bat
```

程序运行后将生成：

```text
data/                       # 自动生成的训练影像、标签和测试影像
outputs/tiny_segnet.pth     # 最终模型权重
outputs/prediction.png      # 测试影像的黑白预测图
```

合成数据只用于讲解代码结构，不能代表真实遥感任务的模型精度。

## 验证状态

已于 2026-08-11 完成端到端运行，实际环境如下：

- Python 3.13.1；
- PyTorch 2.12.1+cu126；
- torchvision 0.27.1+cu126；
- Rasterio 1.5.1；
- NumPy 2.5.2；
- Pillow 12.3.0；
- NVIDIA GeForce RTX 2070，8 GB，驱动 591.86；
- `torch.cuda.is_available()` 为 `True`，CUDA 张量位于 `cuda:0`。

实际生成 12 张训练影像、12 张标签和 1 张测试影像。30 轮训练的平均 loss 从 `0.7055` 降至 `0.0809`；模型权重保存与加载成功，输出的 `prediction.png` 为 256×256，并同时包含 `0` 和 `255` 两种像元值。

示例版本核对于 2026-08-11。若 PyTorch 官方安装命令已经变化，应先更新 `requirements-torch-cu126.txt`，再运行示例。

pip 镜像配置参考：[清华大学开源软件镜像站 PyPI 使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。
