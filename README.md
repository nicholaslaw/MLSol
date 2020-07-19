# A Multilabel Oversampling Method Implemented in Python

Paper: https://arxiv.org/pdf/1905.00609.pdf by Bin Liu and Grigorios Tsoumakas

# Installation
1. docker
```
docker-compose up
```

2. pip
```
pip install . # install package
pip install -r requirements.txt # install jupyter and pytest for testing
```

3. shell script
```
./setup.sh docker # this is essentially step (1)

./setup.sh pip # use this instead if want step (2)
```

# Getting Started
```
import numpy as np
from MLSol import MLSol

sampler = MLSol()
X = np.random.normal(size=(1000, 3))
Y = np.random.randint(2, size=(1000, 5))
X, Y = sampler.oversample(X, Y, 0.1, 5) # generate 10% more instances with 5 nearest neighbors
print(X.shape, Y.shape) # (1100, 3) (1100, 5)
```

# Jupyter Notebook Server
To set up a notebook server, follow step 1 or 3 of Installation and assuming default settings are applied, head to http://localhost:8889/tree to view existing or create new notebooks to perform experiments with the module. Password would be password.

# Testing Your Alterations
```
pytest tests/test_MLSol.py
```
or
```
./test.sh
```