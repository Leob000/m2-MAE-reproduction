from torchvision import transforms

# ImageNet statistics are standard for pre-trained models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_interpolation(interpolation):
    """Converts interpolation string or int to InterpolationMode."""
    if isinstance(interpolation, str):
        if interpolation.lower() == "bicubic":
            return transforms.InterpolationMode.BICUBIC
        if interpolation.lower() == "bilinear":
            return transforms.InterpolationMode.BILINEAR
    return interpolation


def get_pretrain_transforms(
    img_size=64,
    crop_min_scale=0.2,
    interpolation=3,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """Returns transforms for MAE pre-training.

    Args:
        img_size (int): Target image size.
        crop_min_scale (float): Minimum scale for RandomResizedCrop.
        interpolation (int): Interpolation mode (3=BICUBIC).
        mean (tuple): Normalization mean.
        std (tuple): Normalization std.

    Returns:
        transforms.Compose: Composition of transforms.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(crop_min_scale, 1.0),
                interpolation=interpolation,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def get_classification_transforms(
    img_size=64,
    is_train=True,
    crop_min_scale=0.08,
    interpolation=3,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """Returns transforms for linear probing / fine-tuning.

    Args:
        img_size (int): Target image size.
        is_train (bool): Whether to return training or validation transforms.
        crop_min_scale (float): Minimum scale for RandomResizedCrop (train only).
        interpolation (int/str): Interpolation mode.
        mean (tuple): Normalization mean.
        std (tuple): Normalization std.

    Returns:
        transforms.Compose: Composition of transforms.
    """
    interp = _get_interpolation(interpolation)
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    img_size,
                    scale=(crop_min_scale, 1.0),
                    interpolation=interp,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    # Standard validation transforms: resize then center crop
    return transforms.Compose(
        [
            transforms.Resize(
                int(img_size * 1.15), interpolation=interp
            ),  # Slight over-resize
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
