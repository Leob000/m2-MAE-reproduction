from torchvision import transforms

# ImageNet statistics are standard for pre-trained models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def get_val_transforms(
    img_size=64, interpolation=3, mean=IMAGENET_MEAN, std=IMAGENET_STD
):
    """Returns transforms for validation/linear probing.

    Args:
        img_size (int): Target image size.
        interpolation (int): Interpolation mode (3=BICUBIC).
        mean (tuple): Normalization mean.
        std (tuple): Normalization std.

    Returns:
        transforms.Compose: Composition of transforms.
    """
    return transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
