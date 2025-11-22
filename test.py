# tests/test_collab.py
import numpy as np
from collab_llm_predictor import collaborative_predict

class DummyModel:
    def __init__(self, probs):
        self._probs = probs
    def to(self, device): return self
    def eval(self): return self
    def __call__(self, x):
        import torch
        return torch.tensor([self._probs], dtype=torch.float32)

def test_collab_mock():
    model = DummyModel([0.4, 0.39, 0.21])
    class_names = ["COVID", "Normal", "Viral Pneumonia"]
    res = collaborative_predict("tests/sample.jpg", model, class_names, device="cpu", image_size=32, conf_threshold=0.6)
    assert "final_label" in res
    assert isinstance(res["probs"], list)
