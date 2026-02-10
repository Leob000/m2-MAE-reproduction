import logging
import os
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
    correct_1 = 0
    correct_5 = 0
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

        # Top-1 accuracy
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct_1 += predicted.eq(labels).sum().item()

        # Top-5 accuracy
        _, top5 = logits.topk(5, 1, True, True)
        correct_5 += top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()

    num_batches = (
        len(dataloader) if max_steps is None else min(len(dataloader), max_steps)
    )
    return (
        loss_total / num_batches,
        100.0 * correct_1 / total,
        100.0 * correct_5 / total,
    )


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
        mode_name = "finetune" if cfg.train.get("finetune", False) else "linprobe"

        # Try to extract epoch from checkpoint name for better naming
        ckpt_name = "scratch"
        if hasattr(cfg, "pretrained_checkpoint") and cfg.pretrained_checkpoint:
            ckpt_base = os.path.basename(cfg.pretrained_checkpoint)
            # checkpoint-1099.pth -> 1099
            ckpt_name = ckpt_base.replace("checkpoint-", "").replace(".pth", "")

        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            config=wandb_config,
            name=f"{mode_name}-{ckpt_name}",
        )

    # Ensure output directory exists
    os.makedirs(cfg.paths.output_dir, exist_ok=True)

    # Instantiate the pre-trained encoder (MAE)
    # We use the same model config but we will only use the encoder part
    mae_model = hydra.utils.instantiate(cfg.model)

    # Load pre-trained weights if provided
    if hasattr(cfg, "pretrained_checkpoint") and cfg.pretrained_checkpoint:
        logger.info(f"Loading pre-trained weights from {cfg.pretrained_checkpoint}")
        checkpoint = torch.load(
            cfg.pretrained_checkpoint, map_location="cpu", weights_only=False
        )
        # Load state dict with strict=False to ignore decoder weights
        msg = mae_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(
            f"Checkpoint loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}"
        )

    # Create the Classifier wrapper
    finetune = cfg.train.get("finetune", False)
    model = ViTClassifier(
        encoder=mae_model,
        num_classes=cfg.dataset.num_classes,
        embed_dim=cfg.model.embed_dim,
        finetune=finetune,
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

    # Optimizer
    if finetune:
        # Optimize all parameters for fine-tuning
        params = model.parameters()
        logger.info("Fine-tuning: optimizing all parameters")
    else:
        # Only optimize the head for linear probing
        params = model.head.parameters()
        logger.info("Linear Probing: optimizing only the classification head")

    optimizer = torch.optim.AdamW(
        params,
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
        if not finetune:
            # Freeze encoder explicitly for linear probing
            model.encoder.eval()

        train_correct_1 = 0
        train_correct_5 = 0
        train_total = 0

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

            # Calculate training accuracy
            with torch.no_grad():
                _, predicted = logits.max(1)
                train_total += labels.size(0)
                train_correct_1 += predicted.eq(labels).sum().item()
                _, top5 = logits.topk(5, 1, True, True)
                train_correct_5 += (
                    top5.eq(labels.view(-1, 1).expand_as(top5)).sum().item()
                )

            if i % cfg.train.log_interval == 0:
                train_acc1 = 100.0 * train_correct_1 / train_total
                logger.info(
                    f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f} | Acc1: {train_acc1:.2f}% | LR: {lr:.2e}"
                )
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/acc1": train_acc1,
                        "train/acc5": 100.0 * train_correct_5 / train_total,
                        "train/lr": lr,
                        "train/epoch": epoch + i / steps_per_epoch,
                    }
                )

        # Evaluation
        val_loss, val_acc1, val_acc5 = evaluate(
            model, val_loader, device, max_steps=cfg.train.max_steps
        )
        logger.info(
            f"Epoch {epoch} | Val Loss: {val_loss:.4f} | Val Acc1: {val_acc1:.2f}% | Val Acc5: {val_acc5:.2f}%"
        )

        if val_acc1 > best_acc:
            best_acc = val_acc1
            if wandb.run is not None:
                wandb.run.summary["best_val_acc1"] = best_acc
            ckpt_path = os.path.join(cfg.paths.output_dir, "checkpoint-best-clf.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc1,
                },
                ckpt_path,
            )

        wandb.log(
            {
                "val/loss": val_loss,
                "val/acc1": val_acc1,
                "val/acc5": val_acc5,
                "epoch": epoch,
            }
        )

    wandb.finish()


if __name__ == "__main__":
    main()
