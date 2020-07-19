import pytest
import numpy as np
from MLSol import MLSol

def test_MLSol():
    sampler = MLSol()
    X = np.random.normal(size=(1000, 3))
    Y = np.random.randint(2, size=(1000, 5))
    X, Y = sampler.oversample(X, Y, 0.1, 5)

    # Test for increase in number of samples
    assert X.shape == (1100, 3)
    assert Y.shape == (1100, 5)

    # Test for C having no entries outside range [0, 1]
    assert len(sampler.C[(sampler.C < 0) | (sampler.C > 1)]) == 0