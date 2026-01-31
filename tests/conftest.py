import pytest
import torch


@pytest.fixture(scope="session")
def device():
    """Fixture to provide the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def vit_nano_config():
    """Smallest possible ViT config for sub-second testing."""
    return {
        "image_size": 32,
        "patch_size": 4,
        "num_classes": 10,
        "dim": 64,
        "depth": 2,
        "heads": 2,
        "mlp_dim": 128,
        "channels": 3,
        "dropout": 0.1,
        "emb_dropout": 0.1,
    }
