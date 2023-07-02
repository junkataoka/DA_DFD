import pytest
from src.helper import compute_weights
import torch

def test_compute_weights():
    
    features = torch.randn(10, 100)
    targets = torch.randint(0, 10, (10,))
    indices1 = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    cen = torch.randn(10, 100)
    indices2 = torch.tensor([0, 5, 2, 3, 4, 1, 6, 7, 8, 9])
    a1 = compute_weights(features, targets, indices1, cen) 
    a2 = compute_weights(features, targets, indices2, cen)
    assert torch.allclose(a1, a2) is False
