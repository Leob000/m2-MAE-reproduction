# Implementation adapted from: https://github.com/facebookresearch/mae
# Original License: CC-BY-NC 4.0

import torch


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Main entry point to build 2D sine-cosine positional embeddings.

    Args:
        embed_dim: The total dimension of the embedding (e.g., 256).
        grid_size: The number of patches along one side (e.g., 16 for a 16x16 grid).
        cls_token: If True, prepends a zero-vector embedding for the [CLS] token.

    Returns:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    """
    # 1. Create a grid of coordinates (y, x) for every patch.
    # grid_h represents the row index (y), grid_w represents the column index (x).
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)

    # 2. Build the coordinate maps.
    # indexing='xy' means grid[0] will be columns (X) and grid[1] will be rows (Y).
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)  # Shape: [2, grid_size, grid_size]

    # 3. Reshape to add a singleton dimension for the projection function.
    # New shape: [2, 1, grid_size, grid_size]
    grid = grid.reshape([2, 1, grid_size, grid_size])

    # 4. Generate embeddings from the coordinate grid.
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    # 5. Handle CLS token if necessary.
    # The original paper prepends a 0-embedding for the CLS token which doesn't have a spatial position.
    if cls_token:
        pos_embed = torch.cat(
            [torch.zeros([1, embed_dim], dtype=pos_embed.dtype), pos_embed], dim=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Computes 2D embeddings by concatenating independent 1D embeddings for X and Y.

    Args:
        embed_dim: Total dimension. Half is used for height, half for width.
        grid: [2, 1, grid_size, grid_size] coordinate grid.
    """
    assert embed_dim % 2 == 0

    # grid[0] is the X coordinate map (width)
    # grid[1] is the Y coordinate map (height)

    # Use half of the dimensions to encode Y (height)
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # Result: (H*W, D/2)

    # Use half of the dimensions to encode X (width)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # Result: (H*W, D/2)

    # Concatenate spatial components to form the final 2D embedding.
    emb = torch.cat([emb_h, emb_w], dim=1)  # Result: (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """The core math engine for sinusoidal embeddings.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        embed_dim: Output dimension for this component.
        pos: A grid of positions to be encoded (size M).
    """
    assert embed_dim % 2 == 0

    # 1. Create the frequency spectrum (omega).
    # We create d/2 frequencies because each frequency produces both a sine and a cosine.
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # Shape: (D/2,)

    # 2. Project positions onto the frequency spectrum using an outer product.
    pos = pos.reshape(-1)  # Flatten spatial grid to sequence of positions (M,)
    out = torch.einsum("m,d->md", pos, omega)  # Shape: (M, D/2)

    # 3. Compute sine and cosine values.
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    # 4. Concatenate sine and cosine features.
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # Shape: (M, D)
    return emb
