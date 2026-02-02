# Implementation adapted from: https://github.com/facebookresearch/mae
# Original License: CC-BY-NC 4.0

from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from m2_ovo_mae.models.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone.

    This class implements the Masked Autoencoder (MAE) architecture using a Vision
    Transformer (ViT) as the encoder and a lightweight Transformer decoder. It
    handles the random masking of input patches, encoding the visible patches,
    and reconstructing the masked patches.

    Attributes:
        patch_embed (PatchEmbed): The patch embedding layer.
        cls_token (nn.Parameter): The learnable classification token.
        pos_embed (nn.Parameter): The learnable positional embeddings.
        blocks (nn.ModuleList): The list of encoder Transformer blocks.
        norm (nn.Module): The encoder normalization layer.
        decoder_embed (nn.Linear): The linear projection to decoder embedding dimension.
        mask_token (nn.Parameter): The learnable mask token.
        decoder_pos_embed (nn.Parameter): The decoder positional embeddings.
        decoder_blocks (nn.ModuleList): The list of decoder Transformer blocks.
        decoder_norm (nn.Module): The decoder normalization layer.
        decoder_pred (nn.Linear): The final projection to patch pixels.
        norm_pix_loss (bool): Whether to normalize pixel loss.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
    ):
        """Initializes the MaskedAutoencoderViT.

        Args:
            img_size (int, optional): Input image size.
            patch_size (int, optional): Patch size.
            in_chans (int, optional): Number of input channels.
            embed_dim (int, optional): Encoder embedding dimension.
            depth (int, optional): Depth of encoder.
            num_heads (int, optional): Number of attention heads in encoder.
            decoder_embed_dim (int, optional): Decoder embedding dimension.
            decoder_depth (int, optional): Depth of decoder.
            decoder_num_heads (int, optional): Number of attention heads in decoder.
            mlp_ratio (float, optional): MLP expansion ratio.
            norm_layer (nn.Module, optional): Normalization layer.
            norm_pix_loss (bool, optional): Whether to normalize target pixels for loss.
        """
        super().__init__()

        ### MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        ### MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights for pos_embed, patch_embed, cls_token, and mask_token."""
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(decoder_pos_embed.unsqueeze(0))

        # Initialize patch_embed projection using Xavier uniform
        # (standard for Linear layers).
        # We treat the Conv2d projection as a flattened linear operation.
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize CLS and Mask tokens with a standard normal distribution (std=0.02).
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Recursively apply _init_weights to all nn.Linear and nn.LayerNorm submodules.
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for Linear and LayerNorm layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Splits images into non-overlapping patches and flattens them.

        Args:
            imgs (torch.Tensor): Images tensor of shape (N, 3, H, W).

        Returns:
            torch.Tensor: Flattened patches of shape (N, L, patch_size**2 * 3),
                where L is the number of patches.
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        # 1. Split H and W into h, p and w, p respectively.
        # Shape: (N, 3, h, p, w, p)
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))

        # 2. Reorder to group pixels of the same patch together.
        # n: batch, c: channel, h: grid_row, p: patch_row, w: grid_col, q: patch_col
        # Target: (batch, grid_row, grid_col, patch_row, patch_col, channel)
        x = torch.einsum("nchpwq->nhwpqc", x)

        # 3. Flatten into (batch, num_patches, patch_data)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """Reconstructs images from a sequence of flattened patches.

        Args:
            x (torch.Tensor): Flattened patches of shape (N, L, patch_size**2 * 3).

        Returns:
            torch.Tensor: Reconstructed images of shape (N, 3, H, W).
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        # 1. Reshape back to grid and patch dimensions.
        # Shape: (N, h, w, p, p, 3)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))

        # 2. Reorder to put channels and spatial dimensions back in standard order.
        # Target: (batch, channel, grid_row, patch_row, grid_col, patch_col)
        x = torch.einsum("nhwpqc->nchpwq", x)

        # 3. Merge grid and patch dimensions into full H and W.
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by argsort random noise.

        Args:
            x (torch.Tensor): Input sequence tensor of shape (N, L, D).
            mask_ratio (float): The ratio of patches to mask (remove).

        Returns:
            tuple:
                - x_masked (torch.Tensor): Masked sequence (N, L_keep, D).
                - mask (torch.Tensor): Binary mask (N, L), 0 is keep, 1 is remove.
                - ids_restore (torch.Tensor): Indices to restore original order (N, L).
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input images of shape (N, 3, H, W).
            mask_ratio (float): Masking ratio.

        Returns:
            tuple:
                - x (torch.Tensor): Encoded visible patches (N, L_keep, D).
                - mask (torch.Tensor): Binary mask (N, L).
                - ids_restore (torch.Tensor): Indices to restore order (N, L).
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Latent representation from encoder (N, L_keep + 1, D).
            ids_restore (torch.Tensor): Indices to restore original order (N, L).

        Returns:
            torch.Tensor: Decoded predictions (N, L, p*p*3).
        """
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """Calculates the MAE reconstruction loss.

        Args:
            imgs (torch.Tensor): Original images (N, 3, H, W).
            pred (torch.Tensor): Predicted patches (N, L, p*p*3).
            mask (torch.Tensor): Binary mask (N, L), 0 is keep, 1 is remove.

        Returns:
            torch.Tensor: The mean squared error loss on masked patches.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        """Forward pass of the MAE.

        Args:
            imgs (torch.Tensor): Input images (N, 3, H, W).
            mask_ratio (float, optional): Masking ratio.

        Returns:
            tuple:
                - loss (torch.Tensor): Reconstruction loss.
                - pred (torch.Tensor): Predicted patches (N, L, p*p*3).
                - mask (torch.Tensor): Binary mask (N, L).
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_lite_patch4_dec128d2b(**kwargs):
    """Constructs a ViT-Lite MAE model for Tiny ImageNet.

    This configuration roughly follows the ViT-Lite settings from Charisoudis et al. (2022).
    It uses a 64x64 input size, patch size 4, 7 encoder blocks, and 2 decoder blocks.

    Args:
        **kwargs: Additional arguments passed to MaskedAutoencoderViT.

    Returns:
        MaskedAutoencoderViT: The configured MAE model.
    """
    model = MaskedAutoencoderViT(
        img_size=64,
        patch_size=4,
        embed_dim=256,
        depth=7,
        num_heads=4,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# Recommended arch for Tiny ImageNet reproduction
mae_vit_lite_patch4 = mae_vit_lite_patch4_dec128d2b

if __name__ == "__main__":
    model = mae_vit_lite_patch4()
    imgs = torch.randn(2, 3, 64, 64)
    # model.random_masking(imgs, mask_ratio=0.75)
    loss, pred, mask = model(imgs, mask_ratio=0.75)
    print("loss:", loss.item())
    print("pred shape:", pred.shape)
    print("mask shape:", mask.shape)
