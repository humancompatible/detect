import numpy as np
from random import randrange
import scipy.optimize as optimize


def lin_prog_feas(
    hist1: np.ndarray,
    hist2: np.ndarray,
    delta: float,
    num_samples: float = 1.0,
) -> int:
    """Specifies a number of samples as a fraction of the total
    histogram bins and checks whether all the sampled bins satisfy
    
    `|hist1 - hist2| <= delta`

    Args:
        hist1 (np.ndarray): 1-D array (or (n,1) column vector) of histogram bin densities for the full dataset.
        hist2 (np.ndarray): 1-D array (or (n,1) column vector) of histogram bin densities for the subgroup.
        delta (float): Threshold for the absolute difference `|hist1 - hist2|`.
        num_samples (float): Fraction of total bins to sample.
            The function draws int(num_samples * (len(hist1) - 1)) random samples.

    Returns:
        int: Status code from `scipy.optimize.linprog`. A status of 0 indicates
             the constraints are feasible (i.e., `|hist1 - hist2| <= delta` for all
             sampled bins); other codes signal infeasibility or solver errors.
    """
    h1_raw = np.asarray(hist1, dtype=float)
    h2_raw = np.asarray(hist2, dtype=float)

    def _is_vector(x: np.ndarray) -> bool:
        return x.ndim == 1 or (x.ndim == 2 and 1 in x.shape)

    if not _is_vector(h1_raw) or not _is_vector(h2_raw):
        raise ValueError(f"histograms must be 1-D or (n,1)/(1,n); got {h1_raw.shape} and {h2_raw.shape}")

    # Normalize to 1-D
    h1 = h1_raw.reshape(-1)
    h2 = h2_raw.reshape(-1)

    rand_lst1 = []
    rand_lst2 = []
    if num_samples != 1:
        for _ in range(0, int(num_samples * (h1.shape[0] - 1))):
            i = randrange(0, h1.shape[0] - 1)
            rand_lst1.append(float(h1[i]))
            rand_lst2.append(float(h2[i]))
        rand_arr1 = np.expand_dims(np.array(rand_lst1), axis=1)
        rand_arr2 = np.expand_dims(np.array(rand_lst2), axis=1)
    else: # = Case in which no sampling occurs, whole histograms are compared
        rand_arr1 = np.expand_dims(h1,axis=1)
        rand_arr2 = np.expand_dims(h2,axis=1)

    # We are not interested in the optimization itself, but in the
    # feasibility of the problem, therefore the coefficient in the
    # objective function is set to 0 and the only variable (x_0) is
    # fixed at 1
    c = 0
    x0_bounds = (1, 1)

    # Accomodate for the + & - signs of the absolute value in
    # |r_a1 - r_a2| <= delta
    A_ub = np.vstack((rand_arr1, -rand_arr1))
    b_ub = np.vstack((delta + rand_arr2, delta - rand_arr2))

    res = optimize.linprog(c, A_ub, b_ub, bounds=[x0_bounds])
    return res.status
