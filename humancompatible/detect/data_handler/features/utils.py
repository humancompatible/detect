import numpy as np

from ..types import CategValue, OneDimData
from .Binary import Binary
from .Categorical import Categorical
from .Contiguous import Contiguous
from .Feature import Feature
from .Mixed import Mixed


def make_feature(
    data: OneDimData,
    feat_name: str | None,
    categ_vals: list[CategValue] | None,
    real_bounds: list[CategValue] | None,
    ordered: bool,
    discrete: bool,
    monotone: bool,
    modifiable: bool,
) -> Feature:
    """
    Factory that creates the appropriate Feature subclass for a single column.

    The returned class depends on whether the feature is categorical, binary, mixed,
    or continuous, and whether a predefined list of allowed categorical values is given.

    Args:
        data: One feature column (NumPy array or pandas Series).
        feat_name: Feature name (may be ``None``).
        categ_vals: Allowed categorical values. ``None`` means continuous.
        real_bounds: Optional bounds for continuous/mixed features.
        ordered: Whether categorical values should be treated as ordered (if applicable).
        discrete: Whether a continuous feature should be treated as discrete.
        monotone: Whether monotonicity constraints apply.
        modifiable: Whether the feature is modifiable.

    Returns:
        A ``humancompatible.detect.data_handler.features.Feature`` instance
        (one of Binary/Categorical/Contiguous/Mixed).
    """
    if categ_vals is None:
        return Contiguous(
            data,
            feat_name,
            bounds=real_bounds,
            discrete=discrete,
            monotone=monotone,
            modifiable=modifiable,
        )
    else:
        if len(categ_vals) > 0:  # if predefined mapping exists
            if np.any(~np.isin(data, categ_vals)):
                # if there are non-categorical values
                return Mixed(
                    data,
                    categ_vals,
                    name=feat_name,
                    bounds=real_bounds,
                    monotone=monotone,
                    modifiable=modifiable,
                )
            elif len(categ_vals) > 2:
                return Categorical(
                    data,
                    categ_vals,
                    name=feat_name,
                    monotone=monotone,
                    modifiable=modifiable,
                    ordering=categ_vals if ordered else None,
                )
            else:
                return Binary(
                    data,
                    categ_vals,
                    name=feat_name,
                    monotone=monotone,
                    modifiable=modifiable,
                )
        else:
            # fully categorical without pre-specified valuess
            if len(np.unique(data)) > 2:
                return Categorical(
                    data, name=feat_name, monotone=monotone, modifiable=modifiable
                )
            else:
                return Binary(
                    data, name=feat_name, monotone=monotone, modifiable=modifiable
                )
