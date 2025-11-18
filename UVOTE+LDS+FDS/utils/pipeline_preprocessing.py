from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np

"""
This script provides utility classes for preprocessing lists of pandas DataFrames,
primarily those containing time series data, for use in sklearn pipelines.

Classes:
- ApplyThreshold: Filters rows in DataFrames based on a threshold in a specified
  column. It preserves the integrity of time series by splitting a DataFrame into
  multiple DataFrames if the threshold condition creates discontinuities, ensuring
  that resulting sequences meet a minimum length.
- CreateConcatDataset: Transforms a list of DataFrames into a PyTorch
  `ConcatDataset`. Each DataFrame is individually wrapped by a specified dataset
  class before concatenation, preventing data from different original DataFrames
  from being mixed within a single dataset sequence.
- SkScalerWrapper: A wrapper for scikit-learn scalers (e.g., StandardScaler,
  MinMaxScaler) to enable their use with lists of DataFrames. It supports
  `partial_fit` for incremental learning across multiple DataFrames and applies
  the transformation DataFrame by DataFrame.
- _ConcatDataFrames: A utility transformer that concatenates a list of DataFrames
  into a single DataFrame using a MultiIndex. The first level of the MultiIndex
  identifies the original DataFrame. This is primarily used to adapt list-based
  DataFrame processing to scikit-learn's single DataFrame input convention within
  a pipeline.
- _SeparateDataFrames: A utility transformer that performs the reverse operation of
  `_ConcatDataFrames`. It splits a single DataFrame with a MultiIndex (where the
  first level indicates distinct original DataFrames) back into a list of
  DataFrames. This is used to revert from scikit-learn's single DataFrame output
  to a list-based format.
"""

class ApplyThreshold(BaseEstimator, TransformerMixin):
    """
    Transformer that applies a threshold to a specific column of each DataFrame in a list of DataFrames.

    Parameters:
    by (str): The name of the column to apply the threshold to.
    threshold (float): The threshold value.
    seq_length (int, optional): The minimum sequence length to keep. Defaults to 1.
    threshold_as_upper_bound (bool, optional): If True, keeps rows where the column value is less than the threshold. If False, keeps rows where the column value is greater than the threshold. Defaults to False.
    """

    def __init__(self, by, threshold, seq_length=1, threshold_as_upper_bound=False):
        self.by = by
        self.threshold = threshold
        self.seq_length = seq_length
        self.threshold_as_upper_bound = threshold_as_upper_bound

    def fit(self, X, y=None):
        """
        Fit method for compatibility with sklearn API. Does nothing.

        Parameters:
        X: Ignored.
        y: Ignored.
        """
        return self

    def _threshold_condition(self, df):
        """
        Helper method that returns a boolean Series where True indicates the rows of df where the threshold condition is met.

        Parameters:
        df (pd.DataFrame): The DataFrame to apply the threshold condition to.
        """
        if self.threshold_as_upper_bound:
            return df[self.by] < self.threshold
        else:
            return df[self.by] > self.threshold

    def _process_group(self, group):
        """
        Helper method that returns the group if its length is greater than seq_length, and None otherwise.

        Parameters:
        group (pd.DataFrame): The group of rows to process.
        """
        if len(group) > self.seq_length:
            return group

    def transform(self, dflist):
        """
        Applies the threshold to each DataFrame in dflist.

        Parameters:
        dflist (list of pd.DataFrame): The list of DataFrames to transform.
        """
        new_dflist = []
        for df in dflist:
            df_mask = self._threshold_condition(df)
            groups = df[df_mask].groupby((~df_mask).cumsum())
            new_data = [
                self._process_group(group)
                for _, group in groups
                if self._process_group(group) is not None
            ]
            new_dflist.extend(new_data)

        return new_dflist


class CreateConcatDataset(BaseEstimator, TransformerMixin):
    """
    Transformer that creates a ConcatDataset from a list of DataFrames.

    Parameters:
    dataset_class (type): The class to use for the dataset.
    kwargs (dict): Additional arguments to pass to the dataset_class.
    """

    def __init__(self, dataset_class, **kwargs):
        self.dataset_class = dataset_class
        self.dataset_args = kwargs

    def fit(self, X, y=None):
        """
        Fit method for compatibility with sklearn API. Does nothing.

        Parameters:
        X: Ignored.
        y: Ignored.
        """
        return self

    def transform(self, dflist):
        """
        Transforms a list of DataFrames into a ConcatDataset.
        For each DataFrame in dflist, a new instance of dataset_class
        is created with the DataFrame as the first argument and the
        kwargs as additional arguments. These datasets are then
        concatenated into a single ConcatDataset.

        Parameters:
        dflist (list of pd.DataFrame): The list of DataFrames to transform.

        Returns:
        ConcatDataset: The concatenated dataset.
        """
        dslist = []
        for df in dflist:
            dslist.append(self.dataset_class(df, **self.dataset_args))
        datasets = ConcatDataset(dslist)
        return datasets


class SkScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, scaler, columns=None):
        if hasattr(scaler, "partial_fit"):
            self.scaler = scaler.set_output(transform="pandas")
        else:
            raise ValueError(
                "scaler must have partial_fit method to be used with the bagpipe package"
            )
        if columns is not None and not isinstance(columns, list):
            columns = [columns]
        self.columns = columns

    def fit(self, dflist, y=None):
        """
        Fit the scaler to the data. The scaler is fit incrementally
        on each DataFrame in the list.

        Parameters:
        dflist (list of pd.DataFrame): The list of DataFrames to fit the scaler on.
        y: Ignored.

        Returns:
        self: Returns the instance itself.
        """
        self.scaler._reset()
        for df in dflist:
            if self.columns is None:
                self.scaler.partial_fit(df)
            else:
                self.scaler.partial_fit(df[self.columns])
        return self

    def transform(self, dflist):
        """
        Scale the data. The scaler is applied to each DataFrame
        in the list. If columns were specified in the constructor,
        only those columns are scaled. Otherwise, all columns are scaled.

        Parameters:
        dflist (list of pd.DataFrame): The list of DataFrames to scale.

        Returns:
        list of pd.DataFrame: The list of scaled DataFrames.
        """
        new_dflist = []
        for df in dflist:
            df_copy = df.copy()
            if self.columns is not None:
                df_copy[self.columns] = self.scaler.transform(df[self.columns])
            else:
                df_copy = self.scaler.transform(df)
            new_dflist.append(df_copy)
        return new_dflist


class _ConcatDataFrames(BaseEstimator, TransformerMixin):
    """
    A transformer that concatenates a list of DataFrames
    into a single DataFrame. This class is used to translate from
    the bagpipe methods to sklearn's method within a pipline.
    The resulting DataFrame has a multi-index, where the first level
    corresponds to the original DataFrame in the list.

    This transformer does not need to be fit, as it does
    not learn anything from the data.
    """

    def fit(self, X, y=None):
        """
        Fit method for compatibility with sklearn API. Does nothing.

        Parameters:
        X: Ignored.
        y: Ignored.
        """
        return self

    def transform(self, dflist):
        """
        Concatenates a list of DataFrames into a single DataFrame.
        The resulting DataFrame has a multi-index, where the first
        level corresponds to the original DataFrame in the list.

        Parameters:
        dflist (list of pd.DataFrame): The list of DataFrames to concatenate.

        Returns:
        pd.DataFrame: The concatenated DataFrame.
        """
        return pd.concat(dflist, keys=np.arange(0, len(dflist), 1))


class _SeparateDataFrames(BaseEstimator, TransformerMixin):
    """
    A transformer that separates a DataFrame with a multi-index into a list of DataFrames.
    This class is used in a pipeline to translate from sklearn methods back to bagpipe methods.
    Each resulting DataFrame corresponds to one level of the original multi-index.

    This transformer does not need to be fit, as it does not learn anything from the data.
    """

    def fit(self, X, y=None):
        """
        Fit method for compatibility with sklearn API. Does nothing.

        Parameters:
        X: Ignored.
        y: Ignored.
        """
        return self

    def transform(self, df):
        """
        Separates a DataFrame with a multi-index into a list of DataFrames.
        Each resulting DataFrame corresponds to one level of the original
        multi-index.

        Parameters:
        df (pd.DataFrame): The DataFrame to separate.

        Returns:
        list of pd.DataFrame: The list of separated DataFrames.
        """
        return [df.xs(i) for i in df.index.get_level_values(0).unique().to_list()]