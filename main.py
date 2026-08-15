"""GPU 版最小训练与预测流程。"""

from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import RemoteSensingDataset
from model import TinySegNet


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
BATCH_SIZE = 4
EPOCHS = 30
NUM_WORKERS = 0
SEED = 42


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 NVIDIA GPU，请先检查 PyTorch CUDA 环境。")
    return torch.device("cuda:0")


def create_dataloaders() -> tuple[DataLoader, DataLoader]:
    train_dataset = RemoteSensingDataset(
        DATA_DIR / "train" / "images",
        DATA_DIR / "train" / "labels",
    )
    val_dataset = RemoteSensingDataset(
        DATA_DIR / "val" / "images",
        DATA_DIR / "val" / "labels",
    )

    # 训练集每轮打乱顺序，避免模型反复看到固定的样本排列。
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    # 验证阶段不更新参数，固定顺序更便于复核样本和输出。
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    sample_count = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, labels)

        # PyTorch 默认累积梯度；必须在本批次反向传播前清除上一批次的梯度。
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        sample_count += batch_size

    return total_loss / sample_count


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    sample_count = 0

    # 验证只评估当前参数，不构建反向传播所需的计算图，也不更新参数。
    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = loss_fn(logits, labels)

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            sample_count += batch_size

    return total_loss / sample_count


def train_model(device: torch.device) -> Path:
    train_loader, val_loader = create_dataloaders()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = TinySegNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    first_images, first_labels = next(iter(train_loader))
    print("单个批次影像形状：", tuple(first_images.shape))
    print("单个批次标签形状：", tuple(first_labels.shape))

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "best_tiny_segnet.pth"
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存验证损失最低的参数，而不是默认采用最后一轮参数。
            torch.save(model.state_dict(), model_path)
            marker = " <- 保存最佳权重"

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}{marker}"
        )

    print(f"最佳模型权重：{model_path}")
    return model_path


def predict(device: torch.device, model_path: Path) -> Path:
    model = TinySegNet().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    test_path = DATA_DIR / "test" / "images" / "test_00.tif"
    with rasterio.open(test_path) as src:
        image = src.read().astype("float32") / 255.0

    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        prediction = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    prediction_png = (prediction * 255).astype(np.uint8)
    output_path = OUTPUT_DIR / "prediction.png"
    Image.fromarray(prediction_png).save(output_path)

    print("测试影像形状：", tuple(image_tensor.shape))
    print("模型输出形状：", tuple(logits.shape))
    print(f"预测图已保存：{output_path}")
    return output_path


def main() -> None:
    device = require_cuda()
    print("当前 GPU：", torch.cuda.get_device_name(device))

    model_path = train_model(device)
    predict(device, model_path)


if __name__ == "__main__":
    main()
