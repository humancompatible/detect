import numpy as np
from typing import Any
from .binarizer import Binarizer
from .utils import lin_prog_feas


def compute_l_inf(
    X: np.ndarray,
    y: np.ndarray,
    binarizer: Binarizer,
    feature_involved: str,
    subgroup_to_check: Any,
    delta: float,
):
    """Computes the l-infinity distance between two multidimensional histograms.
    Typically, the first one comes from the whole dataset considered,
    while the second, from a particular subgroup of a protected attribute.

    Args:
        X (np.ndarray): Protected-attribute slice of the dataset (same rows as `y`).
        y (np.ndarray): Boolean target vector.
        binarizer (Binarizer): The very `Binarizer` instance used to encode `X`/`y`.
        feature_involved (str): Column name of the protected feature whose subgroup
                                is being checked.
        subgroup_to_check (Any): Refers to the particular subgroup of the protected attribute.
        delta (float): Threshold for the L-infinity norm between the two histograms.

    Returns:
        Informs whether the two histograms compared are within the input threshold.
        Delta in the l-infinity norm.
    
    Raises:
        ValueError: If `delta` is not positive.
        KeyError: If `feature_involved` is not in the binarizer's feature names.
        KeyError: If `subgroup_to_check` is not a valid value for the feature.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")

    if feature_involved not in binarizer.data_handler.feature_names:
        raise KeyError(f"Feature '{feature_involved}' not in protected set")
    
    X_bin = binarizer.data_handler.encode(X, one_hot=False)
    y_bin = binarizer.encode_y(y)

    feat_idx = binarizer.data_handler.feature_names.index(feature_involved)
    feature = binarizer.data_handler.features[feat_idx]

    try:
        subgroup_code = feature.value_mapping[subgroup_to_check]
    except KeyError as e:
        allowed = list(feature.value_mapping.keys())
        raise KeyError(f"{subgroup_to_check!r} not a valid value "
                       f"for '{feature_involved}'. Allowed: {allowed}") from e

    # Retain only the instances with a positive target outcome -> X_bin_pos
    X_bin_pos = X_bin[y_bin == 1]

    # Filter instances of the (potentially) discriminated subgroup -> discr
    discr = X_bin_pos[X_bin_pos[:, feat_idx] == subgroup_code]

    # Create array with the dataset feature values (to create histograms) and
    # get number of encoded subgroups per feature (required for binning)
    bins = []
    columns_all = np.empty(X_bin_pos.shape[0], )
    columns_discr = np.empty(discr.shape[0], )

    for i in range(X_bin_pos.shape[1]):
        if i != feat_idx:
            bins.append(int(X_bin_pos[:, i].max() + 1))
            columns_all = np.vstack((columns_all, X_bin_pos[:, i]))
            columns_discr = np.vstack((columns_discr, discr[:, i]))

    columns_all = columns_all[1:, :]
    columns_discr = columns_discr[1:, :]

    # "Histogramisation"
    all_hist, _ = np.histogramdd(columns_all.T, bins=bins, density=True)
    discr_hist, _ = np.histogramdd(columns_discr.T, bins=bins, density=True)

    # Reshaping
    dim = 1
    for e in all_hist.shape:
        dim *= e

    all_rsh = all_hist.reshape(dim, 1)
    discr_rsh = discr_hist.reshape(dim, 1)

    status = lin_prog_feas(all_rsh, discr_rsh, delta=delta)
    return status, delta
