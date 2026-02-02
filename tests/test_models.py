import torch

from m2_ovo_mae.models.mae import mae_vit_lite_patch4


def test_mae_vit_lite_shapes():
    """Verify the output shapes of ViT-Lite MAE forward pass."""
    # Tiny ImageNet resolution is 64x64
    img_size = 64
    patch_size = 4
    num_patches = (img_size // patch_size) ** 2  # (64//4)**2 = 16*16 = 256

    model = mae_vit_lite_patch4()

    # Dummy batch: (B, C, H, W)
    batch_size = 2
    imgs = torch.randn(batch_size, 3, img_size, img_size)

    loss, pred, mask = model(imgs, mask_ratio=0.75)

    # Check shapes
    # pred should be (B, L, patch_size**2 * 3) = (2, 256, 4*4*3) = (2, 256, 48)
    assert pred.shape == (batch_size, num_patches, patch_size**2 * 3)

    # mask should be (B, L) = (2, 256)
    assert mask.shape == (batch_size, num_patches)

    # Loss should be a scalar
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_mae_vit_lite_masking_ratio():
    """Verify that the masking ratio correctly determines the number of masked patches."""
    img_size = 64
    patch_size = 4
    num_patches = (img_size // patch_size) ** 2

    model = mae_vit_lite_patch4()
    batch_size = 1
    imgs = torch.randn(batch_size, 3, img_size, img_size)

    mask_ratio = 0.75
    _, _, mask = model(imgs, mask_ratio=mask_ratio)

    # Check that the number of masked patches matches the ratio
    # 0 is keep, 1 is remove
    num_masked = mask.sum().item()
    expected_masked = int(num_patches * mask_ratio)
    assert num_masked == expected_masked


def test_mae_vit_lite_determinism():
    """Verify that the masking logic is deterministic with a fixed seed."""
    # Test that fixed seed produces same masking
    img_size = 64
    model = mae_vit_lite_patch4()
    imgs = torch.randn(1, 3, img_size, img_size)

    torch.manual_seed(42)
    _, _, mask1 = model(imgs, mask_ratio=0.75)

    torch.manual_seed(42)
    _, _, mask2 = model(imgs, mask_ratio=0.75)

    assert torch.all(mask1 == mask2)

    # Different seed should produce different mask
    torch.manual_seed(43)
    _, _, mask3 = model(imgs, mask_ratio=0.75)

    assert not torch.all(mask1 == mask3)


def test_mae_patchify_unpatchify():
    """Verify that patchify and unpatchify are inverse operations."""
    img_size = 64
    model = mae_vit_lite_patch4()

    # Create a structured image to make errors obvious (e.g., gradient)
    imgs = torch.randn(2, 3, img_size, img_size)

    patches = model.patchify(imgs)
    reconstructed = model.unpatchify(patches)

    assert reconstructed.shape == imgs.shape
    assert torch.allclose(imgs, reconstructed, atol=1e-6)


def test_mae_norm_pix_loss():
    """Verify that pixel normalization affects the loss value."""
    img_size = 64
    imgs = torch.randn(2, 3, img_size, img_size)

    model_no_norm = mae_vit_lite_patch4(norm_pix_loss=False)
    model_norm = mae_vit_lite_patch4(norm_pix_loss=True)

    # Use same seed for masking
    torch.manual_seed(42)
    loss_no_norm, _, _ = model_no_norm(imgs)

    torch.manual_seed(42)
    loss_norm, _, _ = model_norm(imgs)

    assert loss_no_norm != loss_norm
