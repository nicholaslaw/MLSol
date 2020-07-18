import numpy as np
from MLSol import MLSol


sampler = MLSol()
X = np.random.normal(size=(1000, 3))
Y = np.random.randint(2, size=(1000, 5))
X, Y = sampler.oversample(X, Y, 0.1, 5)
print(X.shape, Y.shape)