from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from ..data_handler import DataHandler
from ..data_handler.features import Binary, Categorical, Contiguous, Feature, Mixed
from ..data_handler.types import CategValue, DataLike, OneDimData

BinValue = float | tuple[float, float] | CategValue | list[CategValue] | bool


class Operation(Enum):
    EQ = "="
    NE = "!="
    LE = "<="
    LT = "<"
    GE = ">="
    GT = ">"
    IN = "in"
    NOT_IN = "not in"
    BETWEEN = "between"
    OUTSIDE = "outside"

    # TODO move the perform to the Bin class
    @classmethod
    def perform(
        cls, op: Operation, vals: np.ndarray[int | float | str], reference: BinValue
    ) -> np.ndarray[bool]:
        if op == Operation.EQ:
            return vals == reference
        elif op == Operation.NE:
            return vals != reference
        elif op == Operation.LE:
            return vals <= reference
        elif op == Operation.LT:
            return vals < reference
        elif op == Operation.GE:
            return vals >= reference
        elif op == Operation.GT:
            return vals > reference
        elif op == Operation.IN:
            result = np.zeros_like(vals, dtype=bool)
            for r in reference:
                result |= vals == r
            return result
        elif op == Operation.NOT_IN:
            return ~Operation.perform(Operation.IN, vals, reference)
        elif op == Operation.BETWEEN:
            return (vals >= reference[0]) & (vals < reference[1])
        elif op == Operation.OUTSIDE:
            return ~Operation.perform(Operation.BETWEEN, vals, reference)
        else:
            raise NotImplementedError(f"Operation {op} is not implemented")

    @classmethod
    def negated(cls, op) -> Operation:
        return {
            Operation.EQ: Operation.NE,
            Operation.NE: Operation.EQ,
            Operation.LE: Operation.GT,
            Operation.LT: Operation.GE,
            Operation.GE: Operation.LT,
            Operation.GT: Operation.LE,
            Operation.IN: Operation.NOT_IN,
            Operation.NOT_IN: Operation.IN,
            Operation.BETWEEN: Operation.OUTSIDE,
            Operation.OUTSIDE: Operation.BETWEEN,
        }[op]


class Bin:
    def __init__(self, feature: Feature, operation: Operation, value: BinValue):
        self.feature = feature
        self.operation = operation
        self.value = value

    def negate_self(self):
        if isinstance(self.feature, Binary):
            vals = list(self.feature.value_mapping.keys())
            negated_value = vals[0] if vals[1] == self.value else vals[1]
            return Bin(self.feature, self.operation, negated_value)
        else:
            return Bin(self.feature, Operation.negated(self.operation), self.value)

    def evaluate(self, values: np.ndarray[int | float | str]) -> np.ndarray[bool]:
        return Operation.perform(self.operation, values, self.value)

    def __repr__(self):
        return f"Bin({repr(self.feature)}, {repr(self.operation)}, {repr(self.value)})"

    def __str__(self):
        return f"{str(self.feature)} {self.operation.value} {str(self.value)}"

    def __eq__(self, other):
        return (
            self.feature == other.feature
            and self.operation == other.operation
            and self.value == other.value
        )


class Binarizer:
    """Handles binarizing the dataset"""

    # TODO add specific options for binarization of categoricals (only positive and custom sets) and continuous (custom bins - i.e. quantiles)
    def __init__(
        self,
        data_handler: DataHandler,
        target_positive_vals: list[CategValue] | None = None,
    ):
        self.__original_dhandler = data_handler

        binarized_features: list[list[Bin]] = []
        binarized_negations: list[list[Bin]] = []
        for feature in data_handler.features:
            if isinstance(feature, Contiguous):
                binarizations = []
                negations = []
                minval, maxval = feature.bounds
                # to make the last bin include the max value
                eps = (maxval - minval) / 10000
                n_bins = 10
                prev = minval
                for curr in np.linspace(minval, maxval + eps, n_bins + 1)[1:]:
                    bounds = (prev, curr)
                    binarizations.append(Bin(feature, Operation.BETWEEN, bounds))
                    negations.append(Bin(feature, Operation.OUTSIDE, bounds))
                    prev = curr
                binarized_features.append(binarizations)
                binarized_negations.append(negations)
            elif isinstance(feature, Mixed):
                raise NotImplementedError("Mixed features are not yet implemented")
            elif isinstance(feature, Binary):
                inv_map = {i: v for v, i in feature.value_mapping.items()}
                binarized_features.append([Bin(feature, Operation.EQ, inv_map[1])])
                binarized_negations.append([Bin(feature, Operation.EQ, inv_map[0])])
            elif isinstance(feature, Categorical):
                binarizations = []
                negations = []
                for value in feature.orig_vals:
                    binarizations.append(Bin(feature, Operation.EQ, value))
                    negations.append(Bin(feature, Operation.NE, value))
                binarized_features.append(binarizations)
                binarized_negations.append(negations)
            else:
                raise ValueError("Unsupported feature type")

        # TARGET
        target = data_handler.target_feature
        if isinstance(target, Binary):
            inv_map = {i: v for v, i in target.value_mapping.items()}
            self.binarized_target = Bin(target, Operation.EQ, inv_map[1])
            self.binarized_target_neg = Bin(target, Operation.EQ, inv_map[0])
        elif isinstance(target, Categorical) and (target_positive_vals is not None):
            self.binarized_target = Bin(target, Operation.IN, target_positive_vals)
            negative_vals = [
                v for v in target.orig_vals if v not in target_positive_vals
            ]
            self.binarized_target_neg = Bin(target, Operation.IN, negative_vals)
        else:
            raise NotImplementedError(
                "Target feature must be Binary or Categorical with single binarization"
            )

        self.__binarized_features = binarized_features
        self.__binarized_negations = binarized_negations

    def encode(
        self, X: DataLike, include_negations=False, include_binary_negations=False
    ) -> np.ndarray[bool]:
        if isinstance(X, pd.DataFrame):
            X = X.values

        values = []
        for i, binariaztions in enumerate(self.__binarized_features):
            for bin in binariaztions:
                values.append(Operation.perform(bin.operation, X[:, [i]], bin.value))
        if include_negations:
            for i, binariaztions in enumerate(self.__binarized_negations):
                for bin in binariaztions:
                    values.append(
                        Operation.perform(bin.operation, X[:, [i]], bin.value)
                    )
        elif include_binary_negations:
            for i, binariaztions in enumerate(self.__binarized_negations):
                for bin in binariaztions:
                    if isinstance(bin.feature, Binary):
                        values.append(
                            Operation.perform(bin.operation, X[:, [i]], bin.value)
                        )
        return np.hstack(values)

    def encode_y(self, y: OneDimData) -> np.ndarray[bool]:
        if isinstance(y, pd.Series):
            y = y.values
        res = Operation.perform(
            self.binarized_target.operation, y, self.binarized_target.value
        )
        return res.flatten()

    def __feature_name_tuples(self, include_negations, include_binary_negations):
        names = []
        if include_negations:
            feats = self.__binarized_features + self.__binarized_negations
        else:
            feats = [f for f in self.__binarized_features]
            if include_binary_negations:
                for binarization in self.__binarized_negations:
                    if isinstance(binarization[0].feature, Binary):
                        feats.append(binarization)
        for binarization in feats:
            for bin in binarization:
                names.append((bin.feature.name, bin.operation.value, str(bin.value)))
        return names

    def feature_names(
        self, include_negations=False, include_binary_negations=False
    ) -> list[str]:
        return [
            f"{feat} {op} {val}"
            for (feat, op, val) in self.__feature_name_tuples(
                include_negations, include_binary_negations
            )
        ]

    def target_name(self) -> tuple[str, str]:
        bin = self.binarized_target
        positive = f"{bin.feature} {bin.operation.value} {bin.value}"
        bin = self.binarized_target_neg
        negative = f"{bin.feature} {bin.operation.value} {bin.value}"
        return positive, negative

    def multi_index_feats(
        self, include_negations=False, include_binary_negations=False
    ) -> pd.MultiIndex:
        return pd.MultiIndex.from_tuples(
            self.__feature_name_tuples(include_negations, include_binary_negations),
            names=["feature", "operation", "value"],
        )

    def get_bin_encodings(
        self, include_negations=False, include_binary_negations=False, return_flat=True
    ):
        if include_negations:
            feats = self.__binarized_features + self.__binarized_negations
        else:
            feats = [f for f in self.__binarized_features]
            if include_binary_negations:
                for binarization in self.__binarized_negations:
                    if isinstance(binarization[0].feature, Binary):
                        feats.append(binarization)
        if not return_flat:
            return feats
        flat = []
        for binariaztions in feats:
            for bin in binariaztions:
                flat.append(bin)
        return flat

    @property
    def data_handler(self):
        return self.__original_dhandler
