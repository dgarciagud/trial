import pandas as pd
import numpy as np
from skfolio.model_selection import CombinatorialPurgedCV

class SkfolioCombinatorialPurgedCV:
    """
    Generates tuples of train_idx, test_idx pairs using CombinatorialPurgedCV from skfolio.
    Assumes the MultiIndex contains a level specified by 'date_idx'.
    """

    def __init__(self,
                 n_splits=3,
                 n_test_splits=1,
                 train_size=252,
                 test_size=21,
                 purge_size=0,
                 embargo_size=0,
                 date_idx='date',
                 random_state=None):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.train_size = train_size
        self.test_size = test_size
        self.purge_size = purge_size
        self.embargo_size = embargo_size
        self.date_idx = date_idx
        self.random_state = random_state

        self._cv_splitter = CombinatorialPurgedCV(
            n_splits=self.n_splits,
            n_test_splits=self.n_test_splits,
            train_size=self.train_size,
            test_size=self.test_size,
            purge_size=self.purge_size,
            embargo_size=self.embargo_size,
            random_state=self.random_state
        )

    def split(self, X, y=None, groups=None):
        if not isinstance(X, pd.DataFrame) or self.date_idx not in X.index.names:
            raise ValueError(f"X must be a pandas DataFrame with a '{self.date_idx}' level in its MultiIndex.")

        # Create a dummy Series with unique dates as index for skfolio's CV splitter
        # skfolio's CV splitter expects a Series with a DatetimeIndex
        unique_dates = X.index.get_level_values(self.date_idx).unique().sort_values()
        dummy_series = pd.Series(index=unique_dates, data=np.arange(len(unique_dates)))

        # Map original MultiIndex to a simple range index for easier handling
        original_index_map = pd.Series(index=X.index, data=np.arange(len(X)))

        for train_date_indices, test_date_indices in self._cv_splitter.split(dummy_series):
            # Get the actual dates for train and test splits
            train_dates = dummy_series.index[train_date_indices]
            test_dates = dummy_series.index[test_date_indices]

            # Filter the original DataFrame's indices based on these dates
            train_idx = original_index_map[original_index_map.index.get_level_values(self.date_idx).isin(train_dates)].values
            test_idx = original_index_map[original_index_map.index.get_level_values(self.date_idx).isin(test_dates)].values
            
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        # The actual number of splits can be complex for CombinatorialPurgedCV
        # We return the n_splits parameter passed to the constructor
        return self.n_splits * self.n_test_splits # This is a simplification, actual splits can be higher due to combinations
