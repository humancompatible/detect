import logging

import numpy as np
import ot

from binarizer import Bin, Binarizer

logger = logging.getLogger(__name__)


def our_metric(truth: np.ndarray[bool], estimate: np.ndarray[bool]) -> float:
    # trunk-ignore(bandit/B101)
    assert (
        truth.shape == estimate.shape
    ), "Classification and ground truth have different shape"

    if truth.dtype != bool:
        print("assuming truth values are 0, 1")
        truth = truth == 1
    if estimate.dtype != bool:
        print("assuming estimate values are 0, 1")
        estimate = estimate == 1

    return np.abs(np.mean(estimate[truth]) - np.mean(estimate[~truth]))
    # n_pos = np.sum(truth)
    # n_neg = truth.shape[0] - n_pos
    # errors = truth != estimate
    # return 1 - (np.sum(errors[truth]) / n_pos) - (np.sum(errors[~truth]) / n_neg)
    # essentially  1 - np.mean(errors[truth]) - np.mean(errors[~truth])


def accuracy(truth: np.ndarray[bool], estimate: np.ndarray[bool]) -> float:
    # trunk-ignore(bandit/B101)
    assert (
        truth.shape == estimate.shape
    ), "Classification and ground truth have different shape"

    if truth.dtype != bool:
        print("assuming truth values are 0, 1")
        truth = truth == 1
    if estimate.dtype != bool:
        print("assuming estimate values are 0, 1")
        estimate = estimate == 1

    return np.mean(truth == estimate)


def balance_datasets(
    y: np.ndarray[bool], datasets: list[np.ndarray], seed: int
) -> list[np.ndarray]:
    pos = np.sum(y)
    neg = y.shape[0] - pos
    np.random.seed(seed)
    if pos > neg:
        to_drop = np.random.choice(np.where(y)[0], size=(pos - neg,), replace=False)
    elif neg > pos:
        to_drop = np.random.choice(np.where(~y)[0], size=(neg - pos,), replace=False)
    else:
        to_drop = []
    keep_mask = np.ones_like(y, dtype=bool)
    keep_mask[to_drop] = False

    pruned_datasets = [d[keep_mask] for d in datasets]
    return pruned_datasets


def _eval_term(
    term: list[Bin],
    binarizer: Binarizer,
    X_test: np.ndarray[bool],
    binary_negs_only: bool,
):
    mask = np.ones((X_test.shape[0],), dtype=bool)
    if binary_negs_only:
        bin_feats = [
            str(f)
            for f in binarizer.get_bin_encodings(
                include_negations=False, include_binary_negations=True
            )
        ]
    else:
        bin_feats = [
            str(f) for f in binarizer.get_bin_encodings(include_negations=True)
        ]
    for feat in term:
        feat_i = bin_feats.index(str(feat))
        mask &= X_test[:, feat_i]
    return mask


def eval_terms(
    dnf: list[list[Bin]],
    binarizer: Binarizer,
    X_test: np.ndarray[bool],
    binary_negs_only: bool = False,
):
    masks = [_eval_term(term, binarizer, X_test, binary_negs_only) for term in dnf]
    return masks


def print_dnf(
    dnf: list[list[Bin]], binarizer: Binarizer, term_evals: np.ndarray[float]
):
    positive, negative = binarizer.target_name()
    if len(dnf) == 0:
        term_strs = ["False"]
    else:
        term_strs = ["(" + " AND ".join(sorted(map(str, term))) + ")" for term in dnf]
        if "()" in term_strs:
            term_evals = [term_evals[term_strs.index("()")]]
            term_strs = ["True"]
        maxlen = max(map(len, term_strs))
        ordering = np.argsort(term_strs)
        # ordering = np.argsort(-term_evals)
        term_strs = [
            term_strs[i]
            + " " * (maxlen - len(term_strs[i]))
            + f" <-- (term's our objective: {np.round(term_evals[i], 6)})"
            for i in ordering
        ]

    print(
        "IF \n    " + "\n OR ".join(term_strs) + f"\nTHEN\n {positive} ELSE {negative}",
    )


def _tv_recurse(
    set1: np.ndarray[int],
    set2: np.ndarray[int],
    curr_i: int,
    valid_values: list[list[int]],
    vector: np.ndarray[int],
) -> float:
    # computes 2*total variation between set1 and set2
    if curr_i == len(valid_values):
        p1 = np.mean((set1 == vector).all(axis=1))
        p2 = np.mean((set2 == vector).all(axis=1))
        return abs(p1 - p2)
    else:
        tv_sum = 0
        for val in valid_values[curr_i]:
            vector[curr_i] = val
            tv_sum += _tv_recurse(set1, set2, curr_i + 1, valid_values, vector)
        return tv_sum


def total_variation(set1: np.ndarray[int], set2: np.ndarray[int]) -> float:
    all_data = np.concatenate([set1, set2], axis=0)
    valid_values = [np.unique(all_data[:, i]) for i in range(all_data.shape[1])]
    return _tv_recurse(set1, set2, 0, valid_values, np.empty((set1.shape[1],))) / 2


def TV_binarized(X0: np.ndarray[int], X1: np.ndarray[int]) -> float:
    dist = 0
    n0, n1 = X0.shape[0], X1.shape[0]
    X0, counts0 = np.unique(X0, return_counts=True, axis=0)
    X1, counts1 = np.unique(X1, return_counts=True, axis=0)
    unseen_mask = np.ones((X1.shape[0],), dtype=bool)
    for count0, x in zip(counts0, X0):
        count1 = 0
        indices = np.where(np.all(X1 == x, axis=1))[0]
        if indices.shape[0] != 0:
            unseen_mask[indices[0]] = False
            count1 = counts1[indices[0]]
        dist += abs(count0 / n0 - count1 / n1)
    for count1, x in zip(counts1[unseen_mask], X1[unseen_mask]):
        indices = np.where(np.all(X0 == x, axis=1))[0]
        if indices.shape[0] != 0:
            # already accounted for
            logger.warning(f"Seen indices did not contain {indices}")
            continue
        dist += count1 / n1  # prob in the other is 0
    return dist / 2


def wasserstein_distance(
    X0: np.ndarray[float], X1: np.ndarray[float], Wtype: str, true_dimension: int
) -> float:
    n0, n1 = X0.shape[0], X1.shape[0]
    X0, counts0 = np.unique(X0, return_counts=True, axis=0)
    X1, counts1 = np.unique(X1, return_counts=True, axis=0)
    if Wtype == "W1":
        dist_matrix = ot.dist(X0, X1, p=2, metric="euclidean") / np.sqrt(
            2 * true_dimension
        )
    else:
        dist_matrix = ot.dist(X0, X1, p=2, metric="sqeuclidean") / (2 * true_dimension)
    # print(np.max(dist_matrix))
    dist = ot.emd2(counts0 / n0, counts1 / n1, dist_matrix, numItermax=1e6)
    if Wtype == "W2":
        dist = np.sqrt(dist)
    return dist


def _overlap_kernel(X: np.ndarray[float], Y: np.ndarray[float]):
    m = X.shape[0]
    n = Y.shape[0]
    # takes too much memory
    # overlaps = np.stack([X for _ in range(n)]).transpose(1, 0, 2) == np.stack(
    #     [Y for _ in range(m)]
    # )
    # return overlaps.mean(axis=2)
    mean_overlaps = np.empty((m, n))
    for i in range(m):
        mean_overlaps[i] = np.mean(Y == X[i].reshape((1, -1)), axis=1)
    return mean_overlaps


def MMD(X0: np.ndarray[float], X1: np.ndarray[float]) -> float:
    n0, n1 = X0.shape[0], X1.shape[0]
    X0, counts0 = np.unique(X0, return_counts=True, axis=0)
    X1, counts1 = np.unique(X1, return_counts=True, axis=0)

    K00 = _overlap_kernel(X0, X0)
    K00 = K00 * counts0.reshape((-1, 1)) * counts0
    # only remove the computation on the same samples, samples with equivalent values are counted in
    # this assumes kernel returns 1 for the same sample
    K00[np.eye(X0.shape[0], X0.shape[0], dtype=bool)] -= counts0
    K01 = _overlap_kernel(X0, X1)
    K01 = K01 * counts0.reshape((-1, 1)) * counts1
    K11 = _overlap_kernel(X1, X1)
    K11 = K11 * counts1.reshape((-1, 1)) * counts1
    K11[np.eye(X1.shape[0], X1.shape[0], dtype=bool)] -= counts1

    mmd_estimate = (
        K00.sum() / (n0 * (n0 - 1))
        + K11.sum() / (n1 * (n1 - 1))
        - 2 * K01.sum() / (n0 * n1)
    )
    return np.sqrt(mmd_estimate)


def term_hamming_distance(term1: list[Bin], term2: list[Bin]) -> int:
    hd = 0
    for b in term1:
        if b not in term2:
            hd += 1
    hd_rest = len(term2) - (len(term1) - hd)
    return hd + hd_rest


def eval_spsf(
    y_model: np.ndarray[bool],
    ingroup: np.ndarray[bool],
    get_direction: bool = False,
) -> float | tuple[float, bool]:
    p_group = np.mean(ingroup)
    p_pos = np.mean(y_model)
    p_joint = np.mean(ingroup & y_model)

    if get_direction:
        return np.abs(p_group * p_pos - p_joint), p_group * p_pos > p_joint
    return np.abs(p_group * p_pos - p_joint)


def eval_fpsf(
    y_true: np.ndarray[bool],
    y_model: np.ndarray[bool] | np.ndarray[float],
    ingroup: np.ndarray[bool],
    get_direction: bool = False,
) -> float:
    assert y_true.shape == y_model.shape and y_model.shape == ingroup.shape
    p_group = np.mean(ingroup & ~y_true)
    p_pos = np.mean(y_model[~y_true])
    p_joint = np.sum(y_model[ingroup & ~y_true]) / y_true.shape[0]

    if get_direction:
        return np.abs(p_group * p_pos - p_joint), p_group * p_pos > p_joint
    return np.abs(p_group * p_pos - p_joint)
