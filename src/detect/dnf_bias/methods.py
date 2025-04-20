import numpy as np
import pandas as pd

from binarizer import Bin, Binarizer


def test_RIPPER(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    ripper_params: dict = {},
):
    from aix360.algorithms.rule_induction.ripper import RipperExplainer

    bin_feats = binarizer.get_bin_encodings(include_negations=False)
    if X_train.shape[1] != len(bin_feats):
        raise ValueError("Ripper method assumes that negations are NOT included")

    colnames = ["".join(b) for b in binarizer.multi_index_feats()]
    X_pd = pd.DataFrame(X_train, columns=colnames).astype(int)
    X_test_pd = pd.DataFrame(X_test, columns=colnames).astype(int)
    y_pd = pd.Series(y_train, name="target").astype(int)

    if verbose:
        print("RIPPER:")
    ripper = RipperExplainer(**ripper_params)
    ripper.fit(X_pd, y_pd, target_label=1)
    ruleset = ripper.explain()

    # print("\n\nHERE IS THE RULESET")
    # print(ruleset)
    # print("END OF RULESET\n\n")

    def uncover_value(literal):
        var_name = (
            literal.feature.variable_names[0]
            .replace(",", ", ")
            .replace("^", "(")
            .replace("$", ")")
        )
        feat = bin_feats[colnames.index(var_name)]
        if literal.value == 1:
            return feat
        else:
            return feat.negate_self()

    dnf = [
        [uncover_value(literal) for literal in term.predicates]
        for term in ruleset.conjunctions
    ]

    return ripper.predict(X_test_pd) == 1, dnf


def test_BRCG(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    brcg_params: dict = {},
):
    from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG

    bin_feats = binarizer.get_bin_encodings(include_negations=True)
    colnames = binarizer.multi_index_feats(include_negations=True)
    if X_train.shape[1] != len(bin_feats):
        bin_feats = binarizer.get_bin_encodings(
            include_negations=False, include_binary_negations=True
        )
        colnames = binarizer.multi_index_feats(
            include_negations=False, include_binary_negations=True
        )
        if X_train.shape[1] != len(bin_feats):
            raise ValueError("BRCG method assumes that negations are also included")

    X_train_pd = pd.DataFrame(X_train, columns=colnames)
    X_test_pd = pd.DataFrame(X_test, columns=colnames)

    if verbose:
        print("BRCG")
        brcg_params["verbose"] = True
    if "solver" not in brcg_params:
        brcg_params["solver"] = "GUROBI"
    model = BooleanRuleCG(**brcg_params)
    model.fit(X_train_pd, y_train)

    # print("\n\nEXPLANATION")
    # print(model.explain()["rules"])
    # print("END OF EXPLANATION\n\n")

    split_dnf = [term.split(" AND ") for term in model.explain()["rules"]]
    colnames = [" ".join(b) for b in colnames]
    dnf = [
        [bin_feats[colnames.index(literal)] for literal in term] for term in split_dnf
    ]
    return model.predict(X_test_pd) == 1, dnf


def test_dnf_mio(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    dnfmio_params: dict = {"n_terms": 5, "time_limit": 120},
) -> tuple[np.ndarray[bool], list[list[Bin]]]:
    from dnf_mio import DNF_MIO

    bin_feats = binarizer.get_bin_encodings(include_negations=True)
    if X_train.shape[1] != len(bin_feats):
        raise ValueError("DNF via MIO assumes that negations are also included")

    if verbose:
        print("DNF using MIO")
        dnfmio_params["verbose"] = True
    dnf_mio = DNF_MIO()
    res = dnf_mio.find_dnf(X_train, y_train, **dnfmio_params)

    dnf = [[bin_feats[t] for t in term] for term in res]
    tot_mask = np.zeros((X_test.shape[0],), dtype=bool)
    for term in res:
        mask = np.ones((X_test.shape[0],), dtype=bool)
        for feat_i in term:
            mask &= X_test[:, feat_i]
        tot_mask |= mask
    return mask, dnf


def test_one_rule(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    onerule_params: dict = {},
) -> tuple[np.ndarray[bool], list[list[Bin]]]:
    from one_rule import OneRule

    # alternative implementation using LP - but it does not use the 0-1 loss
    # from one_rule_lp import OneRule

    bin_feats = binarizer.get_bin_encodings(include_negations=True)
    if X_train.shape[1] != len(bin_feats):
        raise ValueError("One rule method assumes that negations are also included")

    if verbose:
        print("One Rule (using MIO)")
        onerule_params["verbose"] = True
    onerule = OneRule()
    res = onerule.find_rule(X_train, y_train, **onerule_params)

    term = [bin_feats[r] for r in res]
    mask = np.ones((X_test.shape[0],), dtype=bool)
    for feat_i in res:
        mask &= X_test[:, feat_i]
    return mask, [term]


def test_spsf_mio(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    spsf_params: dict = {},
) -> tuple[np.ndarray[bool], list[list[Bin]]]:
    from spsf_mio import SPSF

    bin_feats = binarizer.get_bin_encodings(include_negations=True)
    if X_train.shape[1] != len(bin_feats):
        raise ValueError("SPSF (mio) method assumes that negations are also included")

    if verbose:
        print("SPSF (using MIO)")
        spsf_params["verbose"] = True
    spsf = SPSF()
    res = spsf.find_rule(X_train, y_train, **spsf_params)

    term = [bin_feats[r] for r in res]
    mask = np.ones((X_test.shape[0],), dtype=bool)
    for feat_i in res:
        mask &= X_test[:, feat_i]
    return mask, [term]


def test_MDSS(
    X_train: np.ndarray[bool],
    y_train: np.ndarray[bool],
    X_test: np.ndarray[bool],
    binarizer: Binarizer,
    verbose: bool = False,
    # trunk-ignore(ruff/B006)
    mdss_params: dict = {},
) -> tuple[np.ndarray[bool], list[list[Bin]]]:
    # made for binary inputs only, one could improve the performance of this for categorical values, possibly
    from aif360.detectors.mdss_detector import bias_scan

    bin_feats = binarizer.get_bin_encodings(include_negations=False)
    if X_train.shape[1] != len(bin_feats):
        raise ValueError("BRCG method assumes that negations are also included")

    fnames = binarizer.feature_names(include_negations=False)
    X_train_pd = pd.DataFrame(X_train, columns=fnames)

    if "alpha" not in mdss_params:
        mdss_params["alpha"] = 0.5

    privileged_subset, _ = bias_scan(
        data=X_train_pd,
        observations=pd.Series(y_train, name="target"),
        expectations=None,
        scoring="BerkJones",
        favorable_value=0,
        overpredicted=True,
        **mdss_params,
    )
    res = [
        (fnames.index(name), vals[0])
        for name, vals in privileged_subset.items()
        if len(vals) == 1
    ]  # more than 1 val means that there are both true and false
    term = [bin_feats[r] if v else bin_feats[r].negate_self() for r, v in res]

    mask = np.ones((X_test.shape[0],), dtype=bool)
    for feat_i, val in res:
        mask &= X_test[:, feat_i] == val

    return mask, [term]


def test_SPSF(
    X_test: np.ndarray[bool],
    y_test: np.ndarray[bool],
    verbose: bool = False,
):
    from gerryfair.model import Auditor

    X_test = pd.DataFrame(X_test)

    # by seting the true ys to 0, this becomes equivalent to SP
    auditor = Auditor(X_test, np.zeros(X_test.shape[0]), "FP")

    [ingroup, fairness_violation] = auditor.audit(y_test)
    if verbose:
        print("Fairness violation of the SPSF subgroup is ", fairness_violation)
        # print(np.unique(X_test.values[ingroup], axis=0))
        # print(np.unique(X_test.values[np.array(ingroup) == 0, :], axis=0))
    return np.array(ingroup) == 1, None
