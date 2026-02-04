import torch

from m2_ovo_mae.models.pos_embed import get_2d_sincos_pos_embed


def test_get_2d_sincos_pos_embed_shape():
    """Verify that the positional embeddings have the correct shape."""
    embed_dim = 128
    grid_size = 8

    # Without CLS token
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
    assert pos_embed.shape == (grid_size * grid_size, embed_dim)

    # With CLS token
    pos_embed_cls = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True)
    assert pos_embed_cls.shape == (grid_size * grid_size + 1, embed_dim)
    # First row (CLS) should be zeros
    assert torch.all(pos_embed_cls[0] == 0)


def test_get_2d_sincos_pos_embed_values():
    """Verify that different grid positions have different embeddings."""
    embed_dim = 64
    grid_size = 4
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)

    # Embeddings should not be identical for different patches
    assert not torch.allclose(pos_embed[0], pos_embed[1])
    assert not torch.allclose(pos_embed[0], pos_embed[grid_size])


def test_get_2d_sincos_pos_embed_invalid_dim():
    """Verify that odd embed_dim raises an error."""
    import pytest

    with pytest.raises(AssertionError):
        get_2d_sincos_pos_embed(127, 8)
