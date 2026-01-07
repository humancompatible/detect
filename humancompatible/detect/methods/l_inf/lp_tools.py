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
        hist1 (np.ndarray): 1-D array of histogram bin densities for the full dataset.
        hist2 (np.ndarray): 1-D array of histogram bin densities for the subgroup.
        delta (float): Threshold for the absolute difference `|hist1 - hist2|`.
        num_samples (float): Fraction of total bins to sample.
            The function draws int(num_samples * (len(hist1) - 1)) random samples.

    Returns:
        int: Status code from `scipy.optimize.linprog`. A status of 0 indicates
             the constraints are feasible (i.e., `|hist1 - hist2| <= delta` for all
             sampled bins); other codes signal infeasibility or solver errors.
    """
    rand_lst1 = []
    rand_lst2 = []

    for _ in range(0, int(num_samples * (hist1.shape[0] - 1))):
        i = randrange(0, hist1.shape[0] - 1)
        rand_lst1.append(float(hist1[i]))
        rand_lst2.append(float(hist2[i]))

    rand_arr1 = np.expand_dims(np.array(rand_lst1), axis=1)
    rand_arr2 = np.expand_dims(np.array(rand_lst2), axis=1)

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
