import numpy as np
from utils import lin_prog_feas


def compute_l_inf(
    X_bin,
    y_bin,
    feature,
    subgroup,
    **method_kwargs
):
    """Computes the l-infinity distance between two multidimensional histograms.
    Tipycally, the first one comes from the whole dataset considered,
    while the second, from a particular subgroup of a protected attribute.

    Args:
        X_bin: Contains all the dataset attributes, but the target attribute
        y_bin: Dataset target attribute
        feature: Identifies the protected attribute
        subgroup: Refers to the particular subgroup of the protected attribute

    Returns:
        Informs whether the two histograms compared are within the input threshold
        Delta in the l-infinity norm.
    """

    # Retain only the instances with a positive target outcome -> X_bin_pos
    X_bin_pos = X_bin[y_bin == 1]

    # Filter instances of the (potentially) discriminated subgroup -> discr
    discr = X_bin_pos[X_bin_pos[:,feature] == subgroup]

    # Create array with the dataset feature values (to create histograms) and
    # get number of encoded subgroups per feature (required for binning)
    bins = []
    columns_all = np.empty(X_bin_pos.shape[0],) # dummy array
    columns_discr = np.empty(discr.shape[0],)   # dummy array
    for i in range(X_bin_pos.shape[1]):
        if i != feature:
            bins.append(int(X_bin_pos[:,i].max()+1))
            columns_all = np.vstack((columns_all,X_bin_pos[:,i]))
            columns_discr = np.vstack((columns_discr,discr[:,i]))
    columns_all = columns_all[1:,:]
    columns_discr = columns_discr[1:,:]

    # "Histogramisation"
    all_hist,edges = np.histogramdd(columns_all.T,bins=bins,density=True)
    discr_hist,edges = np.histogramdd(columns_discr.T,bins=bins,density=True)

    # Reshaping
    dim = 1
    for e in all_hist.shape:
        dim *= e

    all_rsh = all_hist.reshape(dim,1)
    discr_rsh = discr_hist.reshape(dim,1)


    res = lin_prog_feas(all_rsh,discr_rsh,method_kwargs['delta'])

    return res
