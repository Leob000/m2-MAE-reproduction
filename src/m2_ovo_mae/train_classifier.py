import logging
import os
import time
from typing import Any, cast

import hydra
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from m2_ovo_mae.dataset.tiny_imagenet import TinyImageNet
from m2_ovo_mae.dataset.transforms import get_classification_transforms
from m2_ovo_mae.models.classifier import ViTClassifier
from m2_ovo_mae.train_pretrain import adjust_learning_rate

logger = logging.getLogger(__name__)


def get_device():
    """Detects and returns the appropriate device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataloader, device, max_steps=None):
    """Evaluates the model on the validation set."""
    model.eval()
    correct = 0
    total = 0
    loss_total = 0
    criterion = nn.CrossEntropyLoss()

    for i, (imgs, labels) in enumerate(dataloader):
        if max_steps is not None and i >= max_steps:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        loss_total += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    num_batches = (
        len(dataloader) if max_steps is None else min(len(dataloader), max_steps)
    )
    return loss_total / num_batches, 100.0 * correct / total


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    """Main training entry point for Linear Probing."""
    # Setup
    torch.manual_seed(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # WandB initialization
    if wandb.run is None:
        wandb_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            config=wandb_config,
            name=f"linprobe-{time.strftime('%Y%m%d-%H%M%S')}",
        )

    # Ensure output directory exists
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # Instantiate the pre-trained encoder (MAE)
    # We use the same model config but we will only use the encoder part
    mae_model = hydra.utils.instantiate(cfg.model)

    # Load pre-trained weights if provided
    if hasattr(cfg, "pretrained_checkpoint") and cfg.pretrained_checkpoint:
        logger.info(f"Loading pre-trained weights from {cfg.pretrained_checkpoint}")
        checkpoint = torch.load(cfg.pretrained_checkpoint, map_location="cpu")
        # Load state dict with strict=False to ignore decoder weights
        msg = mae_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(
            f"Checkpoint loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}"
        )

    # Create the Classifier wrapper
    model = ViTClassifier(
        encoder=mae_model,
        num_classes=cfg.dataset.num_classes,
        embed_dim=cfg.model.embed_dim,
    )
    model.to(device)

    # Dataset & Dataloaders
    train_transform = get_classification_transforms(
        img_size=cfg.dataset.img_size,
        is_train=True,
        use_randaug=cfg.dataset.augmentation.use_randaug,
        randaug_n=cfg.dataset.augmentation.randaug_n,
        randaug_m=cfg.dataset.augmentation.randaug_m,
        crop_min_scale=cfg.dataset.augmentation.crop_min_scale,
        interpolation=cfg.dataset.augmentation.interpolation,
    )
    val_transform = get_classification_transforms(
        img_size=cfg.dataset.img_size,
        is_train=False,
        interpolation=cfg.dataset.augmentation.interpolation,
    )

    train_dataset = TinyImageNet(
        root=cfg.dataset.root, split="train", download=True, transform=train_transform
    )
    val_dataset = TinyImageNet(
        root=cfg.dataset.root, split="val", download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.get("num_workers", cfg.system.num_workers),
        pin_memory=cfg.dataloader.get("pin_memory", cfg.system.pin_memory),
        persistent_workers=cfg.dataloader.get(
            "persistent_workers", cfg.system.persistent_workers
        ),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.get("num_workers", cfg.system.num_workers),
        pin_memory=cfg.dataloader.get("pin_memory", cfg.system.pin_memory),
        persistent_workers=cfg.dataloader.get(
            "persistent_workers", cfg.system.persistent_workers
        ),
    )

    # Optimizer (MAE paper uses LARS or SGD with specific settings for linprobe)
    # We will use AdamW here
    optimizer = torch.optim.AdamW(
        model.head.parameters(),  # Only optimize the head for linear probing
        lr=0,  # Will be set by adjust_learning_rate
        weight_decay=cfg.optimizer.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    total_epochs = cfg.train.epochs
    steps_per_epoch = len(train_loader)
    best_acc = 0.0

    for epoch in range(total_epochs):
        model.train()
        # Freeze encoder explicitly just in case
        model.encoder.eval()

        for i, (imgs, labels) in enumerate(train_loader):
            if cfg.train.max_steps is not None and i >= cfg.train.max_steps:
                break

            # Adjust learning rate per step
            lr = adjust_learning_rate(optimizer, epoch, i, steps_per_epoch, cfg)

            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % cfg.train.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f} | LR: {lr:.2e}"
                )
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/epoch": epoch + i / steps_per_epoch,
                    }
                )

        # Evaluation
        val_loss, val_acc = evaluate(
            model, val_loader, device, max_steps=cfg.train.max_steps
        )
        logger.info(
            f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch})

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(cfg.paths.output_dir, "checkpoint-best-clf.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                ckpt_path,
            )

    wandb.finish()


if __name__ == "__main__":
    main()
