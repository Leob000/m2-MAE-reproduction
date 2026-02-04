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


@torch.no_grad()
def log_reconstructions(model, dataloader, device, num_samples=4):
    """Logs original, masked, and reconstructed images to WandB."""
    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:num_samples].to(device)

    _, pred, mask = model(imgs)

    # Unpatchify predictions
    pred_imgs = model.unpatchify(pred)

    # Visualize the mask by expanding it to patch pixels
    p = model.patch_embed.patch_size[0]
    mask_expanded = (
        mask.unsqueeze(-1)
        .repeat(1, 1, p**2 * 3)
        .reshape(imgs.shape[0], imgs.shape[2] // p, imgs.shape[3] // p, p, p, 3)
    )
    mask_expanded = torch.einsum("nhwpqc->nchpwq", mask_expanded).reshape(imgs.shape)

    # Masked image: 0 where mask=1 (removed)
    masked_imgs = imgs * (1 - mask_expanded)

    # Combined image (reconstruction on masked patches, original on visible)
    combined_imgs = imgs * (1 - mask_expanded) + pred_imgs * mask_expanded

    # Prepare for WandB
    log_dict = {}
    for i in range(num_samples):
        # Denormalize for logging assuming standard ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

        orig = (imgs[i] * std + mean).clamp(0, 1).cpu()
        masked = (masked_imgs[i] * std + mean).clamp(0, 1).cpu()
        recon = (pred_imgs[i] * std + mean).clamp(0, 1).cpu()
        comb = (combined_imgs[i] * std + mean).clamp(0, 1).cpu()

        log_dict[f"viz/sample_{i}"] = [
            wandb.Image(orig, caption="Original"),
            wandb.Image(masked, caption="Masked"),
            wandb.Image(recon, caption="Reconstruction"),
            wandb.Image(comb, caption="Combined"),
        ]

    wandb.log(log_dict, commit=False)
    model.train()


@hydra.main(version_base=None, config_path="../../configs", config_name="train")
def main(cfg: DictConfig):
    """Main training entry point for MAE pre-training."""
    torch.manual_seed(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    if wandb.run is None:
        wandb_config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

        wandb.init(
            project=cfg.wandb.project,
            mode=cfg.wandb.mode,
            config=wandb_config,
            name=f"pretrain-{time.strftime('%Y%m%d-%H%M%S')}",
        )

    model = hydra.utils.instantiate(cfg.model)
    model.to(device)
    logger.info(f"Model instantiated: {cfg.model._target_}")

    transform = get_pretrain_transforms(
        img_size=cfg.dataset.img_size,
        use_rrc=cfg.dataset.augmentation.use_rrc,
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
        num_workers=cfg.dataloader.get("num_workers", cfg.system.num_workers),
        pin_memory=cfg.dataloader.get("pin_memory", cfg.system.pin_memory),
        persistent_workers=cfg.dataloader.get(
            "persistent_workers", cfg.system.persistent_workers
        ),
        drop_last=cfg.dataloader.drop_last,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0,
        betas=tuple(cfg.optimizer.betas),
        weight_decay=cfg.optimizer.weight_decay,
    )

    model.train()
    steps_per_epoch = len(dataloader)
    total_epochs = cfg.train.epochs

    for epoch in range(total_epochs):
        epoch_loss = 0.0
        num_steps = 0

        for i, (imgs, _) in enumerate(dataloader):
            if cfg.train.max_steps is not None and i >= cfg.train.max_steps:
                break

            lr = adjust_learning_rate(optimizer, epoch, i, steps_per_epoch, cfg)

            imgs = imgs.to(device)

            optimizer.zero_grad()
            loss, _, _ = model(imgs, mask_ratio=cfg.train.mask_ratio)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_steps += 1

            if i % cfg.train.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}/{total_epochs} | Step {i}/{steps_per_epoch} | Loss: {batch_loss:.4f} | LR: {lr:.2e} | GradNorm: {grad_norm:.2f}"
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "train/loss": batch_loss,
                            "train/lr": lr,
                            "train/grad_norm": grad_norm,
                            "train/epoch": epoch + i / steps_per_epoch,
                        }
                    )

        # Log average epoch loss
        avg_loss = epoch_loss / num_steps
        logger.info(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f}")
        if wandb.run is not None:
            wandb.log({"epoch/avg_loss": avg_loss, "epoch/num": epoch})

            # Log reconstructions periodically
            if (epoch + 1) % 50 == 0 or epoch == 0:
                log_reconstructions(model, dataloader, device)

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cfg": cfg,
        }

        last_ckpt_path = os.path.join(cfg.paths.output_dir, "checkpoint-last.pth")
        torch.save(checkpoint_data, last_ckpt_path)

        if (epoch + 1) % cfg.train.save_interval == 0 or (epoch + 1) == total_epochs:
            ckpt_path = os.path.join(cfg.paths.output_dir, f"checkpoint-{epoch}.pth")
            torch.save(checkpoint_data, ckpt_path)
            logger.info(f"Saved historical checkpoint to {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
