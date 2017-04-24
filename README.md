# Markov Models

The `Markov_Models` package builds Markov chains from datasets using learning methods in `scikit-learn`. 

### Installation

    python -m pip install -e . --user

### Usage

    In [1]: import Markov_Models as mm

    In [2]: files = []
    In [3]: files.append(['/path/to/data_set_1/parameter_1', '/path/to/data_set_1/parameter_2'])
    In [4]: files.append(['/path/to/data_set_2/parameter_1', '/path/to/data_set_2/parameter_2'])

    In [5]: data = mm.load.from_CSV(files)
    In [6]: model = mm.Markov_Chain(data, estimator='KMeans')
    In [7]: model.fit(N)
