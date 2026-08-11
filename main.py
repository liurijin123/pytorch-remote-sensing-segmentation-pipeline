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


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 NVIDIA GPU，请先检查 PyTorch CUDA 环境。")
    return torch.device("cuda:0")


def train_model(device: torch.device) -> Path:
    dataset = RemoteSensingDataset(
        DATA_DIR / "train" / "images",
        DATA_DIR / "train" / "labels",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = TinySegNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    first_images, first_labels = next(iter(dataloader))
    print("单个批次影像形状：", tuple(first_images.shape))
    print("单个批次标签形状：", tuple(first_labels.shape))

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} - loss: {average_loss:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / "tiny_segnet.pth"
    torch.save(model.state_dict(), model_path)
    print(f"模型权重已保存：{model_path}")
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
