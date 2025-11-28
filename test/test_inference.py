import torch
from src.model import SimpleCNN

def test_probability_distribution():
    model = SimpleCNN()
    model.eval()

    x = torch.randn(1, 3, 224, 224)
    logits = model(x)

    probs = torch.softmax(logits, dim=1)

    assert probs.shape == (1, 3)
    assert torch.all(probs >= 0)
    assert torch.all(probs <= 1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-3)
