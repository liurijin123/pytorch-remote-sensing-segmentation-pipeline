"""生成用于入门演示的 256×256 三波段 GeoTIFF 和二分类标签。"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
IMAGE_SIZE = 256
TRAIN_SAMPLES = 12
VAL_SAMPLES = 4
SEED = 42


def make_sample(rng: np.random.Generator, index: int) -> tuple[np.ndarray, np.ndarray]:
    """生成一张三波段影像和一张像元值为 0/1 的标签。"""
    rows, cols = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]

    center_x = int(rng.integers(80, 176))
    center_y = int(rng.integers(80, 176))
    radius_x = int(rng.integers(35, 70))
    radius_y = int(rng.integers(25, 60))

    target = (
        ((cols - center_x) / radius_x) ** 2
        + ((rows - center_y) / radius_y) ** 2
        <= 1
    )

    # 偶数样本再加入一个较小目标，让每张图稍有不同。
    if index % 2 == 0:
        small_x = int(rng.integers(35, 90))
        small_y = int(rng.integers(35, 90))
        small_radius = int(rng.integers(12, 25))
        target |= (cols - small_x) ** 2 + (rows - small_y) ** 2 <= small_radius**2

    noise = rng.normal(0, 8, size=(3, IMAGE_SIZE, IMAGE_SIZE))
    image = np.empty((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    # 只为演示构造明显的光谱差异，不代表真实地物光谱。
    image[0] = np.where(target, 55, 150) + noise[0]
    image[1] = np.where(target, 150, 105) + noise[1]
    image[2] = np.where(target, 35, 175) + noise[2]

    image = np.clip(image, 0, 255).astype(np.uint8)
    label = target.astype(np.uint8)
    return image, label


def write_image(path: Path, image: np.ndarray, index: int) -> None:
    transform = from_origin(500000 + index * 3000, 3500000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        count=3,
        dtype="uint8",
        crs="EPSG:32650",
        transform=transform,
    ) as dst:
        dst.write(image)


def write_label(path: Path, label: np.ndarray, index: int) -> None:
    transform = from_origin(500000 + index * 3000, 3500000, 10, 10)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        count=1,
        dtype="uint8",
        crs="EPSG:32650",
        transform=transform,
    ) as dst:
        dst.write(label, 1)


def main() -> None:
    train_images = DATA_DIR / "train" / "images"
    train_labels = DATA_DIR / "train" / "labels"
    val_images = DATA_DIR / "val" / "images"
    val_labels = DATA_DIR / "val" / "labels"
    test_images = DATA_DIR / "test" / "images"

    for directory in (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    for index in range(TRAIN_SAMPLES):
        image, label = make_sample(rng, index)
        name = f"sample_{index:02d}.tif"
        write_image(train_images / name, image, index)
        write_label(train_labels / name, label, index)

    # 验证样本单独生成，不与训练目录共享文件，避免把训练数据用于验证。
    for val_index in range(VAL_SAMPLES):
        sample_index = TRAIN_SAMPLES + val_index
        image, label = make_sample(rng, sample_index)
        name = f"sample_{val_index:02d}.tif"
        write_image(val_images / name, image, sample_index)
        write_label(val_labels / name, label, sample_index)

    test_index = TRAIN_SAMPLES + VAL_SAMPLES
    test_image, _ = make_sample(rng, test_index)
    write_image(test_images / "test_00.tif", test_image, test_index)

    print(f"训练影像：{TRAIN_SAMPLES} 张")
    print(f"验证影像：{VAL_SAMPLES} 张")
    print("测试影像：1 张")
    print(f"数据目录：{DATA_DIR}")


if __name__ == "__main__":
    main()
