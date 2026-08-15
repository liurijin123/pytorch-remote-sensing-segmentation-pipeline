"""最小版遥感语义分割 Dataset。"""

from pathlib import Path

import rasterio
import torch
from torch.utils.data import Dataset


class RemoteSensingDataset(Dataset):
    """把一对同名影像与标签定义为一个可索引的训练样本。"""

    def __init__(self, image_dir: Path, label_dir: Path) -> None:
        self.image_paths = sorted(image_dir.glob("*.tif"))
        self.label_dir = label_dir

        if not self.image_paths:
            raise FileNotFoundError(f"没有在 {image_dir} 中找到 .tif 影像")

        for image_path in self.image_paths:
            label_path = self.label_dir / image_path.name
            if not label_path.exists():
                raise FileNotFoundError(f"影像缺少同名标签：{label_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        label_path = self.label_dir / image_path.name

        with rasterio.open(image_path) as src:
            # 卷积网络需要 [通道, 高, 宽] 的浮点影像，示例值域缩放到 0～1。
            image = src.read().astype("float32") / 255.0

        with rasterio.open(label_path) as src:
            # CrossEntropyLoss 接收 [高, 宽] 的 int64 类别编号，不接收 RGB 标签。
            label = src.read(1).astype("int64")

        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)
        return image_tensor, label_tensor
