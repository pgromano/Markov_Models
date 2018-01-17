from .validation import check_sequence
import numpy as np
from sklearn.model_selection import ShuffleSplit


__all__ = ['ContinuousSequence', 'DiscreteSequence']


class ContinuousSequence(object):
    """ Continuous sequence class to establish standard data format

    Parameters
    ----------
    X : iterable of array-like, shape=(n_sets, n_samples, n_features)
        Continuous time-series data with integer (n_sets) number of trajectory
        datasets of shape (n_samples, n_features). If ContinuousSequence is
        provided, the X attributes are inherited.

    Attributes
    ----------
    values : float, list of numpy.ndarray
        Values of all datasets.
    n_sets : int
        Number of sets of data.
    n_samples : list
        List of number of samples/observations in each set.
    n_features : int
        Number of features/dimensions in dataset. The values of n_features must
        be the same for all sets.
    """

    def __init__(self, X):
        if isinstance(X, ContinuousSequence):
            self.__dict__ = X.__dict__
        else:
            self.values = check_sequence(X, rank=2)
            self.n_sets = len(self.values)
            self.n_samples = [self.values[i].shape[0] for i in range(self.n_sets)]
            assert all([self.values[0].shape[1] == self.values[i].shape[1]
                        for i in range(self.n_sets)]), 'Number of features inconsistent'
            self.n_features = self.values[0].shape[1]

    def concatenate(self, features=None):
        """ Concatenates sequences

        Parameters
        ----------
        features : int, optional
            Feature to concatenate and return. None concatenates all featuress

        Returns
        -------
        seqcat : list of numpy.ndarray
            List of sequences concatenated

        See Also
        --------
            numpy.concatenate
        """

        if (not hasattr(self, '_seqcat')) and (features is None):
            self._seqcat = np.concatenate([self.values[i] for i in range(self.n_sets)])
        if features is None:
            return self._seqcat
        return np.concatenate([self.values[i][:, features] for i in range(self.n_sets)])

    def histogram(self, features=None, bins=10, return_extent=False):
        """ Create histogram of sequences

        Parameters
        ----------
        features : int, optional
            Feature to build histogram. None build histogram along all
            features.
        bins : int or iterable of ints
            Number of bins to generate histogram. If bins is an iterable, it
            must be of length equal to the number of features in sequence.

        Returns
        -------
        seqcat : list of numpy.ndarray
            List of sequences concatenated

        See Also
        --------
            numpy.histogramdd
        """

        his, ext = np.histogramdd(self.concatenate(features), bins=bins)
        if return_extent is True:
            extent = []
            for k in range(len(ext)):
                extent.append(ext[k].min())
                extent.append(ext[k].max())
            return his, extent
        return his

    def sample(self, size=None, features=None, replace=True):
        """ Uniformly sample from sequence data

        Parameters
        ----------
        size : int or list of ints, optional
            Size to sample from sequence
        features : int
            Features to sample along
        replace : bool
            Whether to sample with replacement

        Returns
        -------
        samples : numpy.ndarray
            Sampled values from sequences

        See Also
        --------
            numpy.random.choice
        """

        index = np.random.choice(np.arange(np.sum(self.n_samples)), size, replace)
        if len(size) > 1:
            return self.concatenate(features)[index.ravel()].reshape(size)
        return self.concatenate(features)[index]

    def split(self, train_size=0.75, val_size=None, random_state=None):
        """ Split sequence into cross-validation sets

        Parameters
        ----------
        train_size : float, optional
            Ratio of dataset size to split as training set.
        val_size : float, optional
            Ratio of dataset size to split as validation set. If None then only
            train and test sets are returned.
        random_state : float, optional
            Set seed for random splitting.

        Returns
        -------
        Xtr, Xte, Xva : float, numpy.ndarray
            The training (Xtr) and test (Xte) sets are given in the specified
            proportions. If val_size is not None, then the validation (Xva) set
            is also returned.

        See Also
        --------
            sklearn.model_selection.ShuffleSplit
        """
        assert train_size < 1.0, "Training size must be < 1.0"
        assert train_size > 0.0, "Training size must be > 0.0"
        n_obs = range(np.sum(self.n_samples))
        if val_size is None:
            test_size = 1 - train_size
            assert np.allclose(train_size + test_size, 1), "Total size must equal 1"
            for train_index, test_index in ShuffleSplit(1, test_size, random_state=random_state).split(n_obs):
                pass
            Xcat = self.concatenate()
            return Xcat[train_index], Xcat[test_index]

        test_size = 1 - (train_size + val_size)
        assert np.allclose(train_size + test_size + val_size, 1), "Total size must equal 1"

        for train_index, test_val_index in ShuffleSplit(1, val_size + test_size).split(n_obs):
            n = range(len(test_val_index))
            for i, j in ShuffleSplit(1, val_size / (val_size + test_size)).split(n):
                test_index, val_index = test_val_index[i], test_val_index[j]
        Xcat = self.concatenate()
        return Xcat[train_index], Xcat[test_index], Xcat[val_index]


class DiscreteSequence(object):
    """ Discrete sequence class to establish standard data format

    Parameters
    ----------
    X : iterable of array-like, shape=(n_sets, n_samples)
        Discrete time-series data with integer (n_sets) number of trajectory
        datasets of shape n_samples. If DiscreteSequence is
        provided, the X attributes are inherited.

    Attributes
    ----------
    values : int, list of numpy.ndarray
        Values of all datasets.
    n_sets : int
        Number of sets of data.
    n_samples : list
        List of number of samples/observations in each set.
    """

    def __init__(self, X, n_states=None, labels=None, dtype=None, encoder=None):
        if isinstance(X, DiscreteSequence):
            self.__dict__ = X.__dict__
        else:
            # Check n_states and labels
            if (labels is not None) and (n_states is not None) and (n_states > 0):
                if len(labels) != n_states:
                    raise ValueError("""Length of provided labels must equal
                    n_states.""")

            self.values = check_sequence(X, rank=1, dtype=dtype)
            if encoder is not None:
                self.values = [encoder(val) for val in self.values]

            self.n_sets = len(self.values)
            self.n_samples = [val.shape[0] for val in self.values]

    def concatenate(self):
        """ Concatenates sequences

        Returns
        -------
        seqcat : list of numpy.ndarray
            List of sequences concatenated

        See Also
        --------
            numpy.concatenate
        """

        if not hasattr(self, '_seqcat'):
            self._seqcat = np.concatenate([self.values[i] for i in range(self.n_sets)], 0)
        return self._seqcat

    def counts(self, return_labels=True):
        """ Count the number of unique elements

        Parameters
        ----------
        return_labels : bool
            Whether or not to return labels of unique elements.

        Returns
        -------
        counts : int, numpy.ndarray
            The number of occurances for each unique value within sequences.
        labels : int, numpy.ndarray
            If return_labels is True, then the values of the unique states are
            returned.

        See Also
        --------
            numpy.unique
        """

        return np.unique(self.values, return_counts=return_labels)

    def one_hot(self):
        y_hot = []
        for i in range(self.n_sets):
            y_hot.append(np.zeros((self.n_samples[i], self.n_states)))
            y_hot[i][range(self.n_samples[i]), self.values[i]] = 1
        return y_hot

    def sample(self, size=None, replace=True):
        """ Uniformly sample from sequence data

        Parameters
        ----------
        size : int or list of ints, optional
            Size to sample from sequence
        replace : bool
            Whether to sample with replacement

        Returns
        -------
        samples : numpy.ndarray
            Sampled values from sequences

        See Also
        --------
            numpy.random.choice
        """

        return np.random.choice(self.concatenate(), size, replace)
