from pathlib import Path

import pytest
import torch

from m2_ovo_mae.dataset.tiny_imagenet import TinyImageNet
from m2_ovo_mae.dataset.transforms import get_pretrain_transforms


def create_dummy_tiny_imagenet(root: Path):
    """Generates a dummy Tiny ImageNet dataset structure."""
    dataset_path = root / "tiny-imagenet-200"
    dataset_path.mkdir(parents=True)

    # 1. wnids.txt
    (dataset_path / "wnids.txt").write_text("n01443537\nn01629819")

    # 2. Train structure
    train_cls = dataset_path / "train" / "n01443537" / "images"
    train_cls.mkdir(parents=True)
    (train_cls / "n01443537_0.JPEG").touch()

    # 3. Val structure (Flat initially)
    val_images = dataset_path / "val" / "images"
    val_images.mkdir(parents=True)
    (val_images / "val_0.JPEG").touch()
    (val_images / "val_1.JPEG").touch()

    # 4. Val Annotations
    annotations = (
        "val_0.JPEG\tn01443537\t0\t0\t64\t64\nval_1.JPEG\tn01629819\t0\t0\t64\t64\n"
    )
    (dataset_path / "val" / "val_annotations.txt").write_text(annotations)


@pytest.fixture
def mock_tiny_imagenet(tmp_path):
    """Creates a mock Tiny ImageNet dataset structure."""
    root = tmp_path / "data"
    create_dummy_tiny_imagenet(root)
    return root


def test_tiny_imagenet_structure_and_val_formatting(mock_tiny_imagenet):
    """Test that the dataset loads and formats validation set correctly."""
    # Initialize dataset (should trigger formatting)
    dataset = TinyImageNet(root=mock_tiny_imagenet, split="val", download=False)

    # Check if val was reformatted
    val_path = mock_tiny_imagenet / "tiny-imagenet-200" / "val"
    assert (val_path / "n01443537" / "val_0.JPEG").exists()
    assert (val_path / "n01629819" / "val_1.JPEG").exists()
    assert not (val_path / "images" / "val_0.JPEG").exists()

    # Check dataset len
    assert len(dataset) == 2


def test_tiny_imagenet_train_loading(mock_tiny_imagenet):
    """Test loading training data."""
    dataset = TinyImageNet(root=mock_tiny_imagenet, split="train", download=False)
    assert len(dataset) == 1
    # Check classes
    assert len(dataset.classes) == 1  # Only one class has images in our mock


def test_transforms_shapes():
    """Test that transforms produce correct output shapes."""
    transform = get_pretrain_transforms(img_size=64)
    # Dummy PIL image
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    out = transform(img)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 64, 64)
