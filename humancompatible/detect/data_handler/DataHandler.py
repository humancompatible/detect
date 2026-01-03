from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    Binary,
    Categorical,
    Contiguous,
    Feature,
    Monotonicity,
    make_feature,
)
from .types import CategValue, DataLike, FeatureID, OneDimData


class DataHandler:
    """
    Performs all data processing from a pandas DataFrame/numpy array to a normalized and encoded input
    Expected use is to initialize this with training data and then use it to encode all data.
    Supports mixed encoding, where only some values are categorical
    Normalizes contiguous data to [0, 1] range
    Produces either one-hot encoded data or direct data with mapped categorical data to negative integers
    """

    def __init__(
        self,
        features: list[Feature],
        target: Feature | None = None,
        causal_inc: list[tuple[Feature, Feature]] | None = None,
        greater_than: list[tuple[Feature, Feature]] | None = None,
    ):
        self.__input_features = features
        self.__target_feature = target
        self.__causal_inc = causal_inc if causal_inc is not None else []
        self.__greater_than = greater_than if greater_than is not None else []

    @classmethod
    def from_data(
        cls,
        X: DataLike,
        y: OneDimData | None = None,
        categ_map: dict[FeatureID, list[CategValue]] = {},
        ordered: list[FeatureID] = [],
        bounds_map: dict[FeatureID, tuple[int, int]] = {},
        discrete: list[FeatureID] = [],
        immutable: list[FeatureID] = [],
        monotonicity: dict[FeatureID, Monotonicity] = {},
        # TODO more general causality
        causal_inc: list[tuple[FeatureID, FeatureID]] = [],
        greater_than: list[tuple[FeatureID, FeatureID]] = [],
        regression: bool = False,
        feature_names: list[str] | None = None,
        target_name: str | None = None,
    ) -> DataHandler:
        """
        Construct a DataHandler instance.

        Parameters:
        -----------
            X : array-like (2 dimensional)
                Input features. Shape: (num_samples, num_features)
            y : array-like (1 dimensional)
                Target feature (e.g., labels or regression targets). Shape: (num_samples,)
            categ : dictionary
                Dictionary with indices (or column names for DataFrame) of categorical features as keys
                and a list of unique categorical values as values.
                
                If the list is empty, each unique value of the feature is considered categorical
                If the list is non-empty, but does not cover all values, the feature is considered mixed
            regression : bool
                True if the task is regression, False if y is categorical and task is classification.
            feature_names : optional list of strings
                List of feature names, if None it is recovered from column names if X is a DataFrame
            target_name : optional string
                Name of the target feature, if None it is recovered from X if X is a pandas Series
        """
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = X.columns
            if target_name is not None and y is None:
                print("Taking target values from the X matrix")
                y = X[target_name]
                X = X.drop(columns=target_name)
            X = X.to_numpy()

        if y is not None:
            if target_name is None:
                if isinstance(y, pd.Series):
                    target_name = y.name
                else:
                    target_name = "target"

            if regression:
                target_feature = Contiguous(y, target_name)
            else:
                if len(np.unique(y)) > 2:
                    target_feature = Categorical(y, name=target_name)
                else:
                    target_feature = Binary(y, name=target_name)
                    # TODO make the target values specifiable
        else:
            target_feature = None

        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [None] * n_features
        if len(feature_names) != n_features:
            raise ValueError("Incorrect length of list of feature names.")

        input_features: list[Feature] = []
        # stores lists of categorical values of applicable features, used for mapping to integer values
        for feat_i, feat_name in enumerate(feature_names):
            input_features.append(
                make_feature(
                    X[:, feat_i],
                    feat_name,
                    categ_map.get(feat_name, None),
                    bounds_map.get(feat_name, None),
                    feat_name in ordered,
                    feat_name in discrete,
                    monotone=monotonicity.get(feat_name, Monotonicity.NONE),
                    modifiable=feat_name not in immutable,
                )
            )

        causal_inc = [
            (
                input_features[feature_names.index(i)],
                input_features[feature_names.index(j)],
            )
            for i, j in causal_inc
        ]
        greater_than = [
            (
                input_features[feature_names.index(i)],
                input_features[feature_names.index(j)],
            )
            for i, j in greater_than
        ]
        return DataHandler(input_features, target_feature, causal_inc, greater_than)

    @property
    def causal_inc(self) -> list[tuple[Feature, Feature]]:
        return self.__causal_inc

    @property
    def greater_than(self) -> list[tuple[Feature, Feature]]:
        return self.__greater_than

    @property
    def n_features(self) -> int:
        """Number of features in the input space"""
        return len(self.__input_features)

    @property
    def features(self) -> list[Feature]:
        """List of input features"""
        return self.__input_features

    @property
    def target_feature(self) -> Feature:
        """Target feature"""
        return self.__target_feature

    @property
    def feature_names(self) -> list[str]:
        """List of feature names"""
        return [f.name for f in self.__input_features]

    def encode(
        self, X: DataLike, normalize: bool = True, one_hot: bool = True
    ) -> np.ndarray[np.float64]:
        """
        Encode input features.

        Parameters:
        -----------
        X : array-like
            Input features (data matrix or DataFrame). Shape: (num_samples, num_features)
        normalize : bool, optional
            Whether to normalize the features (default is True).
        one_hot : bool, optional
            Whether to perform one-hot encoding for categorical values (default is True).

        Returns:
        --------
        encoded_X : numpy array
            Encoded input features. Shape: (num_samples, one_hot_features) when one hot encoding is performed, (num_samples, num_features) otherwise
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(X, pd.Series):
            X = X.to_numpy()

        if len(X.shape) == 1:
            Xmat = X.reshape(1, -1)
            return self.encode(Xmat, normalize=normalize, one_hot=one_hot)[0]

        enc = []
        for feat_i, feature in enumerate(self.__input_features):
            enc.append(
                feature.encode(X[:, feat_i], normalize, one_hot).reshape(X.shape[0], -1)
            )

        return np.concatenate(enc, axis=1).astype(np.float64)

    def encode_y(
        self, y: OneDimData, normalize: bool = True, one_hot: bool = True
    ) -> np.ndarray[np.float64]:
        """
        Encode target feature.

        Parameters:
        -----------
        y : array-like
            Target feature (data matrix or DataFrame of labels or regression targets). Shape: (num_samples,)
        normalize : bool, optional
            Whether to normalize the features (default is True).
        one_hot : bool, optional
            Whether to perform one-hot encoding for categorical values (default is True).

        Returns:
        --------
        encoded_y : numpy array
            Encoded target feature. Shape: (num_samples, num_values) for one hot encoding or (num_samples,) otherwise
        """
        return self.__target_feature.encode(y, normalize, one_hot)

    def encode_all(self, X_all: np.ndarray, normalize: bool, one_hot: bool):
        return np.concatenate(
            [
                self.encode(X_all[:, :-1], normalize, one_hot),
                self.encode_y(X_all[:, -1], normalize, one_hot).reshape(-1, 1),
            ],
            axis=1,
        )

    def decode(
        self,
        X: np.ndarray[np.float64],
        denormalize: bool = True,
        encoded_one_hot: bool = True,
        as_dataframe: bool = True,
    ) -> np.ndarray[np.float64]:
        """
        Decode input features.

        Parameters:
        -----------
        X : array-like
            Input data matrix. Shape: (num_samples, num_enc_features)
                where num_enc_features can be higher than num_features, because of one-hot encoding
        denormalize : bool, optional
            Whether to invert the normalization of the features (default is True).
        encoded_one_hot : bool, optional
            Whether the input matrix is one-hot encoded (default is True).
        as_dataframe : bool, optional
            Whether to return a pandas DataFrame or numpy array (default is True - DataFrame).

        Returns:
        --------
        decoded_X : numpy array
            Decoded features in the original format. Shape: (num_samples, num_features)
        """
        if X.shape[0] == 0:
            if as_dataframe:
                return pd.DataFrame([], columns=[f.name for f in self.__input_features])
            return np.empty((0, self.n_features))
        dec = []
        curr_col = 0
        for feature in self.__input_features:
            w = feature.encoding_width(encoded_one_hot)
            dec.append(
                feature.decode(X[:, curr_col : curr_col + w], denormalize, as_dataframe)
            )
            curr_col += w
        if as_dataframe:
            return pd.concat(dec, axis=1)
        return np.concatenate([x.reshape(X.shape[0], -1) for x in dec], axis=1)

    def decode_y(
        self,
        y: np.ndarray[np.float64],
        denormalize: bool = True,
        as_series: bool = True,
    ) -> np.ndarray[np.float64]:
        """
        Decode target feature.

        Parameters:
        -----------
        y : array-like
            Target feature data. Shape: (num_samples,) for general case
                or (num_samples, num_categorical_values) in case of one-hot encoding
        denormalize : bool, optional
            Whether to invert the normalization of the feature (default is True).
        as_series : bool, optional
            Whether to return a pandas Series or numpy array (default is True - Series).

        Returns:
        --------
        decoded_y : numpy array
            Decoded target feature data. Shape: (num_samples,)
        """
        return self.__target_feature.decode(y, denormalize, as_series)

    def encoding_width(self, one_hot: bool) -> int:
        return sum([f.encoding_width(one_hot) for f in self.__input_features])

    def allowed_changes(self, pre_vals, post_vals):
        for f, pre, pos in zip(self.features, pre_vals, post_vals):
            if not f.allowed_change(pre, pos):
                return False

        for cause, effect in self.__causal_inc:
            cause_i = self.features.index(cause)
            pre_cause = cause.encode(pre_vals[cause_i], normalize=False, one_hot=False)
            pos_cause = cause.encode(post_vals[cause_i], normalize=False, one_hot=False)
            if isinstance(cause, Categorical):
                applied = pos_cause in cause.greater_than(pre_cause)
            elif isinstance(cause, Contiguous):
                applied = pos_cause > pre_cause
            else:
                raise ValueError("invalid feature type")
            if applied:
                effect_i = self.features.index(effect)
                pre_effect = effect.encode(
                    pre_vals[effect_i], normalize=False, one_hot=False
                )
                pos_effect = effect.encode(
                    post_vals[effect_i], normalize=False, one_hot=False
                )
                if isinstance(effect, Categorical):
                    if pos_effect not in effect.greater_than(pre_effect):
                        return False
                elif isinstance(effect, Contiguous):
                    if pos_effect <= pre_effect:
                        return False
                else:
                    raise ValueError("invalid feature type")

        for greater, smaller in self.__greater_than:
            if (
                post_vals[self.features.index(smaller)]
                > post_vals[self.features.index(greater)]
            ):
                return False
        return True

    # TODO dalsi nadstavba - datawrapper - ktera si bude pamatovat jestli se slo one-hot, normalizovalo atd
