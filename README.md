# Markov Models

The `Markov_Models` package builds Markov chains from datasets using `scikit-learn` for unsupervised classification. The `Markov_Chain` module can be used to generally manipulate and cluster datasets, whereas the `MSM` module is intended for building Markov State Models and contains analysis tools consistent with the field.

### Installation

    python -m pip install -e . --user

### Usage
A detailed example of generating a Markov State Model (MSM) can be found in [here](./examples/standard_msm.ipynb).

    In [1]: import Markov_Models as mm

    In [2]: files = []
    In [3]: files.append(['/path/to/data_set_1/parameter_1', '/path/to/data_set_1/parameter_2'])
    In [4]: files.append(['/path/to/data_set_2/parameter_1', '/path/to/data_set_2/parameter_2'])

    In [5]: data = mm.load.from_CSV(files)
    In [6]: model = mm.Markov_Chain(data, estimator='KMeans')
    In [7]: model.fit(N)
