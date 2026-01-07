import logging
import numpy as np
from typing import Any
from .lp_tools import lin_prog_feas

from humancompatible.detect.binarizer import Binarizer

logger = logging.getLogger(__name__)


def check_l_inf_gap(
    X: np.ndarray,
    y: np.ndarray,
    binarizer: Binarizer,
    feature_involved: str,
    subgroup_to_check: Any,
    delta: float,
    verbose: int = 1,
) -> bool:
    """
    Test whether a protected subgroup's outcome distribution differs from the
    overall population by **at most** `delta` in the l_inf-norm.

    Args:
        X (np.ndarray): Protected-attribute slice of the dataset (same rows as `y`).
        y (np.ndarray): Boolean target vector.
        binarizer (Binarizer): The binarizer used to encode `X` and `y`.
        feature_involved (str): Name of the protected column whose subgroup is tested.
        subgroup_to_check (Any): Raw value of the subgroup to isolate.
        delta (float): Threshold for the L-infinity norm.
        verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
            2 = all detailed logs (including solver output).

    Returns:
        bool: True if the subgroup histogram is within `delta`; False otherwise.

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
    all_counts, _ = np.histogramdd(columns_all.T, bins=bins, density=False)
    discr_counts, _ = np.histogramdd(columns_discr.T, bins=bins, density=False)

    all_tot = all_counts.sum()
    discr_tot = discr_counts.sum()
    if all_tot == 0 or discr_tot == 0:
        raise ValueError("Zero total counts after filtering; cannot compute ℓ∞.")

    all_hist = all_counts / all_tot
    discr_hist = discr_counts / discr_tot

    # Reshaping
    dim = 1
    for e in all_hist.shape:
        dim *= e

    all_rsh = all_hist.reshape(dim, 1)
    discr_rsh = discr_hist.reshape(dim, 1)

    status = lin_prog_feas(all_rsh, discr_rsh, delta=delta)
    is_within = bool(status == 0)  # 0 = feasible
    if is_within:
        if verbose >= 1: logger.info(f"The most impacted subgroup bias <= {delta}")
    else:
        if verbose >= 1: logger.info(f"The most impacted subgroup bias > {delta}")

    return is_within
