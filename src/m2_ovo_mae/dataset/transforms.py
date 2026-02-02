from torchvision import transforms

# ImageNet statistics are standard for pre-trained models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_pretrain_transforms(img_size=64):
    """Returns transforms for MAE pre-training.

    Args:
        img_size (int): Target image size.

    Returns:
        transforms.Compose: Composition of transforms.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size, scale=(0.2, 1.0), interpolation=3
            ),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(img_size=64):
    """Returns transforms for validation/linear probing.

    Args:
        img_size (int): Target image size.

    Returns:
        transforms.Compose: Composition of transforms.
    """
    return transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=3),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
