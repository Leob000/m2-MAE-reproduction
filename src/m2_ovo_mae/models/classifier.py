import torch
import torch.nn as nn


class ViTClassifier(nn.Module):
    """Linear classifier wrapper for a pre-trained ViT encoder."""

    def __init__(self, encoder, num_classes=200, embed_dim=256):
        """Initializes the ViTClassifier.

        Args:
            encoder (nn.Module): The pre-trained ViT encoder.
            num_classes (int): Number of classification classes.
            embed_dim (int): The embedding dimension of the encoder.
        """
        super().__init__()
        self.encoder = encoder

        self.norm = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): Input images (N, 3, H, W).

        Returns:
            torch.Tensor: Classification logits (N, num_classes).
        """
        # Get frozen features from CLS token
        # For linear probing, encoder is frozen.
        with torch.no_grad():
            features = self.encoder.forward_features(x)

        # Apply norm and linear head
        x = self.norm(features)
        x = self.head(x)
        return x
