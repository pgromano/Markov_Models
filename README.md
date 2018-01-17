# Markov Models

The `Markov_Models` package builds Markov chains from datasets and provides an API built on `scikit-learn` to discretized continuous time-series data through unsupervised classification. Currently only discrete models are supported, however future versions will contain continuous methods.

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

The toolkit is designed to follow the Scikit-Learn API, such that all methods can be `fit` and where appropriate `predict`, `transform`, etc. Current applications of the toolkit involve

1. High Order Markov Chains (`models.MarkovChain`)
1. Markov State Models and Spectral Analysis (`models.MarkovStateModel`)
1. Time Series Coarse Graining (`coarse_grain`)

Discrete time analysis can be evaluated via Markov State Models (`MarkovStateModel`) a first order Markov chain where continuous time-series data is discretized on a complete and disjoint set of states. This model can be generated in two ways 1) where the transition probability matrix is known *a priori* and 2) where the model is fit from discrete sequence data. Continuous time-series data can be discretized using the `cluster` tools, which are discussed at greater lengths within the [Scikit-Learn documentation][1].

##### 1) Predefined Transition Matrix

Here, let's randomly initialize a count matrix where all elements $c_{ij}$ give the number of events observed from state $i$ to $j$ over a lagtime $\tau$ (`lag`).

```python
from Markov_Models.models import MarkovStateModel
import numpy as np

# Generate a pseudo count matrix
np.random.seed(42)
C = np.random.randint(0, 100000, size=(2, 2))

# Create transition matrix by Symmetric estimation
T = C / C.sum(1)[:, None]

# Initialize the Markov chain
model_true = MarkovStateModel(T=T, lag=1)
```

##### 2) Estimate the model

By running a simulation from our ideal model, we can estimate a new one and see how accurately we can reproduce transition probabilities under the influence of sampling/noise. We'll use the same lag time, and we will estimate the transition probabilities using the Prinz maximum likelihood estimation scheme. We see our estimated and ideal models return a low root mean squared error.

```python
# Generate a random discrete sequence
seq = model_true.simulate(1000000, random_state=42)

# Initial and fit the Markov chain
model_est = MarkovStateModel(lag=1, method='Prinz').fit(seq)

# RMSE : 0.00044504602839456724
RMSE = np.sqrt(((model_true.transition_matrix -
                 model_est.transition_matrix)**2).sum() /
                 model_true.n_states**2)

```

[1]: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster
