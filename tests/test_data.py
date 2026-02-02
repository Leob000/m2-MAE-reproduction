import pytest
import torch

from m2_ovo_mae.dataset.tiny_imagenet import TinyImageNet
from m2_ovo_mae.dataset.transforms import get_pretrain_transforms


@pytest.fixture
def mock_tiny_imagenet(tmp_path):
    """Creates a mock Tiny ImageNet dataset structure."""
    root = tmp_path / "data"
    dataset_path = root / "tiny-imagenet-200"
    dataset_path.mkdir(parents=True)

    # Create wnids.txt
    (dataset_path / "wnids.txt").write_text("n01443537\nn01629819")

    # Create Train
    (dataset_path / "train" / "n01443537" / "images").mkdir(parents=True)
    (dataset_path / "train" / "n01443537" / "images" / "n01443537_0.JPEG").touch()

    # Create Val (Flat structure initially)
    val_images = dataset_path / "val" / "images"
    val_images.mkdir(parents=True)
    (val_images / "val_0.JPEG").touch()
    (val_images / "val_1.JPEG").touch()

    # Create Val Annotations
    (dataset_path / "val" / "val_annotations.txt").write_text(
        "val_0.JPEG\tn01443537\t0\t0\t64\t64\nval_1.JPEG\tn01629819\t0\t0\t64\t64\n"
    )

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
