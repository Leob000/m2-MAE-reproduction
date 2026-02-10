import torch
import torch.nn as nn

from m2_ovo_mae.models.classifier import ViTClassifier
from m2_ovo_mae.models.mae import mae_vit_lite_patch4


def test_classifier_shapes():
    """Verify the output shapes of ViTClassifier forward pass."""
    embed_dim = 256
    num_classes = 200
    batch_size = 4
    img_size = 64

    # Instantiate encoder and classifier
    encoder = mae_vit_lite_patch4()
    model = ViTClassifier(encoder, num_classes=num_classes, embed_dim=embed_dim)

    # Dummy input
    imgs = torch.randn(batch_size, 3, img_size, img_size)

    # Forward pass
    logits = model(imgs)

    assert logits.shape == (batch_size, num_classes)
    assert not torch.isnan(logits).any()


def test_classifier_frozen_encoder():
    """Verify that encoder gradients are not tracked during classifier forward pass if desired."""
    # Note: The forward method in ViTClassifier uses torch.no_grad() for encoder
    encoder = mae_vit_lite_patch4()
    model = ViTClassifier(encoder, num_classes=10, embed_dim=256)

    imgs = torch.randn(2, 3, 64, 64)
    imgs.requires_grad = True

    logits = model(imgs)
    loss = logits.sum()
    loss.backward()

    # Since encoder is used within torch.no_grad(), its parameters should not have gradients
    # from this specific forward pass if we only optimized the head.
    # However, in PyTorch, if a param is NOT frozen (requires_grad=True) but used in no_grad,
    # it won't get grads.

    # In ViTClassifier.forward:
    # with torch.no_grad():
    #     features = self.encoder.forward_features(x)

    for _name, param in model.encoder.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0)

    # Head should have gradients
    assert model.head.weight.grad is not None
    assert not torch.all(model.head.weight.grad == 0)


def test_classifier_finetune_gradients():
    """Verify that encoder gradients ARE tracked during fine-tuning forward pass."""
    encoder = mae_vit_lite_patch4()
    model = ViTClassifier(encoder, num_classes=10, embed_dim=256, finetune=True)

    imgs = torch.randn(2, 3, 64, 64)
    imgs.requires_grad = True

    logits = model(imgs)
    loss = logits.sum()
    loss.backward()

    # In fine-tuning mode, encoder parameters SHOULD have gradients
    encoder_has_grads = False
    for _name, param in model.encoder.named_parameters():
        if param.grad is not None and not torch.all(param.grad == 0):
            encoder_has_grads = True
            break

    assert encoder_has_grads, "Encoder should have gradients in fine-tuning mode"

    # Head should also have gradients
    assert model.head.weight.grad is not None
    assert not torch.all(model.head.weight.grad == 0)


def test_classifier_norm():
    """Verify that BatchNorm1d is present in the classifier."""
    encoder = mae_vit_lite_patch4()
    model = ViTClassifier(encoder, num_classes=10, embed_dim=256)

    assert isinstance(model.norm, nn.BatchNorm1d)
    assert model.norm.affine is False
