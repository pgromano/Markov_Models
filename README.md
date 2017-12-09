# Markov Models

The `Markov_Models` package builds Markov chains from datasets and provides an API through `scikit-learn` to discretized continuous time-series data through unsupervised classification. Currently only discrete models are supported, however future versions will contain continuous methods.

### Installation

The current version of `Markov_Models` can be installed via pip, using the following commands.

    git clone https://github.com/pgromano/Markov_Models.git
    cd Markov_Models
    python -m pip install -e

To use the previous version of `Markov_Models`, install from the legacy branch.

    git clone -b legacy https://github.com/pgromano/Markov_Models.git
    cd Markov_Models
    python -m pip install -e

### Usage

The toolkit is designed to follow the Scikit API, such that all methods can be `fit` and where appropriate `predict`, `transform`, etc.

Continuous time-series data can be discretized using the `cluster`, which are discussed at greater lengths within the [Scikit-Learn documentation][1].

The primary model builds a discrete Markov chain from two possible inputs, the first directly from a known transition matrix or the second by fitting a model from discrete time-series sequences.

```python
from Markov_Models.models import MarkovChain
import numpy as np

# Generate a pseudo count matrix
C = np.random.randint(0, 100000, size=(2, 2))

# Create transition matrix by Symmetric estimation
Cs = (C + C.T)/2
T = Cs / Cs.sum(1)[:, None]

# Initialize the Markov chain
model = MarkovChain(T=T)
```

**or**

```python
# Generate a random discrete sequence
seq = np.random.randint(0, 2, size=100000)

# Initial and fit the Markov chain
estimator = MarkovChain().fit(seq)
```

[1]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
