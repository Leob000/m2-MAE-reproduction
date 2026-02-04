import logging
import math
import os
import time
from typing import Any, cast

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from m2_ovo_mae.dataset.tiny_imagenet import TinyImageNet
from m2_ovo_mae.dataset.transforms import get_pretrain_transforms
from m2_ovo_mae.models.mae import MaskedAutoencoderViT  # noqa: F401

logger = logging.getLogger(__name__)


def get_device():
    """Detects and returns the appropriate device (cuda, mps, or cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def adjust_learning_rate(optimizer, epoch, i, steps_per_epoch, cfg):
    """Decays the learning rate with half-cycle cosine after warmup."""
    # Base LR (e.g., 5e-4) * (Effective Batch Size / 256)
    eff_batch_size = cfg.dataloader.batch_size
    base_lr = cfg.optimizer.base_lr * eff_batch_size / 256

    warmup_epochs = cfg.train.warmup_epochs
    total_epochs = cfg.train.epochs

    curr_step = epoch * steps_per_epoch + i
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if curr_step < warmup_steps:
        lr = base_lr * curr_step / warmup_steps
    else:
        lr = (
            base_lr
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi * (curr_step - warmup_steps) / (total_steps - warmup_steps)
                )
            )
        )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    """Main training entry point for MAE pre-training."""
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
            name=f"pretrain-{time.strftime('%Y%m%d-%H%M%S')}",
        )

    # Model
    model = hydra.utils.instantiate(cfg.model)
    model.to(device)
    logger.info(f"Model instantiated: {cfg.model._target_}")

    # Dataset & Dataloader
    transform = get_pretrain_transforms(
        img_size=cfg.dataset.img_size,
        crop_min_scale=cfg.dataset.augmentation.crop_min_scale,
        interpolation=cfg.dataset.augmentation.interpolation,
    )
    dataset = TinyImageNet(
        root=cfg.dataset.root,
        split="train",
        download=True,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.system.num_workers,
        pin_memory=cfg.system.pin_memory,
        persistent_workers=cfg.system.persistent_workers,
        drop_last=cfg.dataloader.drop_last,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0,  # Will be set by adjust_learning_rate
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Training Loop
    model.train()
    steps_per_epoch = len(dataloader)
    total_epochs = cfg.train.epochs

    for epoch in range(total_epochs):
        for i, (imgs, _) in enumerate(dataloader):
            if cfg.train.max_steps is not None and i >= cfg.train.max_steps:
                break

            # Adjust learning rate per step
            lr = adjust_learning_rate(optimizer, epoch, i, steps_per_epoch, cfg)

            imgs = imgs.to(device)

            optimizer.zero_grad()
            loss, _, _ = model(imgs, mask_ratio=cfg.train.mask_ratio)
            loss.backward()
            optimizer.step()

            if i % cfg.train.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}/{total_epochs} | Step {i}/{steps_per_epoch} | Loss: {loss.item():.4f} | LR: {lr:.2e}"
                )
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/epoch": epoch + i / steps_per_epoch,
                    }
                )

        # Save checkpoint periodically
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
        }

        # Save "last" checkpoint every epoch
        last_ckpt_path = os.path.join(cfg.paths.output_dir, "checkpoint-last.pth")
        torch.save(checkpoint_data, last_ckpt_path)

        # Save historical checkpoints at intervals or on the final epoch
        if (epoch + 1) % cfg.train.save_interval == 0 or (epoch + 1) == total_epochs:
            ckpt_path = os.path.join(cfg.paths.output_dir, f"checkpoint-{epoch}.pth")
            torch.save(checkpoint_data, ckpt_path)
            logger.info(f"Saved historical checkpoint to {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
