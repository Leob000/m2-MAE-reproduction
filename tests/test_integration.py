import math

import torch
from hydra import compose, initialize

from m2_ovo_mae.train_pretrain import adjust_learning_rate, get_device


def test_get_device():
    """Verify that get_device returns a torch.device."""
    device = get_device()
    assert isinstance(device, torch.device)


def test_adjust_learning_rate():
    """Verify that the learning rate is correctly adjusted during warmup."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="train")

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(2, 2))], lr=0)

    # Warmup phase
    lr = adjust_learning_rate(optimizer, epoch=0, i=0, steps_per_epoch=100, cfg=cfg)
    assert lr == 0
    assert optimizer.param_groups[0]["lr"] == 0

    # At half warmup (warmup_epochs/2), should be base_lr / 2 (for batch 256)
    half_warmup = cfg.train.warmup_epochs // 2
    lr = adjust_learning_rate(
        optimizer, epoch=half_warmup, i=0, steps_per_epoch=100, cfg=cfg
    )
    expected_lr = cfg.optimizer.base_lr * 0.5
    assert math.isclose(lr, expected_lr, rel_tol=1e-5)

    # Post-warmup (cosine decay)
    # At the very end of training
    lr_end = adjust_learning_rate(
        optimizer, epoch=cfg.train.epochs - 1, i=99, steps_per_epoch=100, cfg=cfg
    )
    # Cosine(pi) = -1. 0.5 * (1 + -1) = 0. Should be close to 0 but not necessarily exactly 0 depending on float precision.
    assert lr_end < expected_lr
    assert lr_end >= 0


def test_train_one_step(tmp_path):
    """Smoke test: Run one pre-training step on dummy data."""
    # 1. Setup dummy dataset structure
    data_root = tmp_path / "data"
    dataset_path = data_root / "tiny-imagenet-200"
    dataset_path.mkdir(parents=True)
    (dataset_path / "wnids.txt").write_text("n01")
    (dataset_path / "train" / "n01" / "images").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (64, 64)).save(
        dataset_path / "train" / "n01" / "images" / "im1.JPEG"
    )
    # Create val directory as well to satisfy TinyImageNet initialization
    (dataset_path / "val" / "images").mkdir(parents=True)
    (dataset_path / "val" / "val_annotations.txt").write_text(
        "im_val.JPEG\tn01\t0\t0\t64\t64\n"
    )
    Image.new("RGB", (64, 64)).save(dataset_path / "val" / "images" / "im_val.JPEG")

    # 2. Setup Hydra config
    with initialize(version_base=None, config_path="../configs"):
        # We just test the compose works, we don't strictly need cfg for the rest
        # of the manual instantiation in this smoke test.
        compose(
            config_name="train",
            overrides=[
                f"paths.project_root={tmp_path}",
                f"paths.data_dir={data_root}",
                "dataloader.batch_size=1",
                "system.num_workers=0",
            ],
        )

    # 3. Instantiate model
    from m2_ovo_mae.models.mae import mae_vit_lite_patch4

    model = mae_vit_lite_patch4()
    device = torch.device("cpu")
    model.to(device)

    # 4. Data
    from m2_ovo_mae.dataset.tiny_imagenet import TinyImageNet
    from m2_ovo_mae.dataset.transforms import get_pretrain_transforms

    transform = get_pretrain_transforms(img_size=64)
    dataset = TinyImageNet(
        root=str(data_root), split="train", download=False, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

    # 5. One Step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    imgs, _ = next(iter(dataloader))
    imgs = imgs.to(device)

    loss, _, _ = model(imgs)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)


def test_classifier_one_step(tmp_path):
    """Smoke test: Run one classification training step on dummy data."""
    # 1. Setup dummy dataset structure
    data_root = tmp_path / "data"
    dataset_path = data_root / "tiny-imagenet-200"
    dataset_path.mkdir(parents=True)
    (dataset_path / "wnids.txt").write_text("n01")
    (dataset_path / "train" / "n01" / "images").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (64, 64)).save(
        dataset_path / "train" / "n01" / "images" / "im1.JPEG"
    )
    Image.new("RGB", (64, 64)).save(
        dataset_path / "train" / "n01" / "images" / "im2.JPEG"
    )
    (dataset_path / "val" / "images").mkdir(parents=True)
    (dataset_path / "val" / "val_annotations.txt").write_text(
        "im_val.JPEG\tn01\t0\t0\t64\t64\n"
    )
    Image.new("RGB", (64, 64)).save(dataset_path / "val" / "images" / "im_val.JPEG")

    # 2. Instantiate models
    from m2_ovo_mae.models.classifier import ViTClassifier
    from m2_ovo_mae.models.mae import mae_vit_lite_patch4

    encoder = mae_vit_lite_patch4()
    model = ViTClassifier(encoder, num_classes=200, embed_dim=256)
    device = torch.device("cpu")
    model.to(device)

    # 3. Data
    from m2_ovo_mae.dataset.tiny_imagenet import TinyImageNet
    from m2_ovo_mae.dataset.transforms import get_classification_transforms

    transform = get_classification_transforms(img_size=64, is_train=True)
    dataset = TinyImageNet(
        root=str(data_root), split="train", download=False, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    # 4. One Step
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    imgs, labels = next(iter(dataloader))
    imgs, labels = imgs.to(device), labels.to(device)

    logits = model(imgs)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
