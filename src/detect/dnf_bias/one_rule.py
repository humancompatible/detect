import logging

import numpy as np
import pyomo.environ as pyo

# ignore assert warnings
# trunk-ignore-all(bandit/B101)


logger = logging.getLogger(__name__)


class OneRule:
    """Implementation of a MIO formulation for finding an optimal conjunction with lowest 0-1 error.
    The formulation is inspired by 1Rule method from http://proceedings.mlr.press/v28/malioutov13.pdf
    """

    def __init__(self) -> None:
        pass

    def _make_int_model(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        # trunk-ignore(ruff/B006)
        feat_init: dict[int, int] = {},
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal conjunction using 0-1 loss.

        Args:
            X (np.ndarray[bool]): input matrix
            y (np.ndarray[bool]): target labels
            feat_init (dict[int, int], optional): Initialization of the conjunction.
                A dictionary containing feature indices as keys and 0/1 values of whether they are used. Defaults to {}.

        Returns:
            pyo.ConcreteModel: The MIO model containing the formulation
        """
        n, d = X.shape
        Xint = np.zeros_like(X, dtype=int)
        Xint[X] = 1

        model = pyo.ConcreteModel()
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        model.use_feat = pyo.Var(model.feat_i, domain=pyo.Binary, initialize=feat_init)
        model.error = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # model.non_empty = pyo.Constraint(
        #     expr=sum(model.use_feat[j] for j in model.feat_i) >= 1
        # )

        # positive - error = 1 if at least one where u_j = 1 has X_j=0
        #     else error = 0
        # negative - error = 1 if all where u_j = 1 have X_j=1 as well
        #     else error = 0

        model.pos = pyo.Constraint(
            model.pos_i,
            model.feat_i,
            rule=lambda m, i, j: (
                m.use_feat[j] - Xint[i, j] * m.use_feat[j] <= m.error[i]
            ),
        )
        model.neg = pyo.Constraint(
            model.neg_i,
            rule=lambda m, i: (
                sum(m.use_feat[j] - Xint[i, j] * m.use_feat[j] for j in m.feat_i)
                + m.error[i]
                >= 1
            ),
        )

        model.obj = pyo.Objective(
            expr=sum(model.error[i] * weights[i] for i in model.all_i),
            sense=pyo.minimize,
        )

        return model

    def _make_abs_model(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        n_min: int,
        # trunk-ignore(ruff/B006)
        feat_init: dict[int, int] = {},
    ) -> pyo.ConcreteModel:
        """Create the Integer Optimiztion formulation to find an optimal conjunction.

        Args:
            X (np.ndarray[bool]): input matrix
            y (np.ndarray[bool]): target labels
            feat_init (dict[int, int], optional): Initialization of the conjunction.
                A dictionary containing feature indices as keys and 0/1 values of whether they are used. Defaults to {}.

        Returns:
            pyo.ConcreteModel: The MIO model containing the formulation
        """
        n, d = X.shape
        Xint = np.zeros_like(X, dtype=int)
        Xint[X] = 1

        model = pyo.ConcreteModel()
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])

        model.use_feat = pyo.Var(model.feat_i, domain=pyo.Binary, initialize=feat_init)
        model.ingroup = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))

        model.force_0 = pyo.Constraint(
            model.all_i,
            model.feat_i,
            rule=lambda m, i, j: (
                m.ingroup[i] <= 1 - (m.use_feat[j] - Xint[i, j] * m.use_feat[j])
            ),
        )
        model.force_1 = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                m.ingroup[i]
                >= 1 - sum(m.use_feat[j] - Xint[i, j] * m.use_feat[j] for j in m.feat_i)
            ),
        )

        model.minimum = pyo.Constraint(
            expr=sum(model.ingroup[i] for i in model.all_i) >= n_min
        )

        model.o = pyo.Var(domain=pyo.NonNegativeReals)
        model.b = pyo.Var(domain=pyo.Binary)
        term1 = sum(model.ingroup[i] * weights[i] for i in model.pos_i)
        term2 = sum(model.ingroup[i] * weights[i] for i in model.neg_i)
        model.abs_obj_u1 = pyo.Constraint(expr=model.o <= term1 - term2 + 2 * model.b)
        model.abs_obj_u2 = pyo.Constraint(
            expr=model.o <= term2 - term1 + 2 * (1 - model.b)
        )
        model.obj = pyo.Objective(
            expr=model.o,
            sense=pyo.maximize,
        )

        return model

    def find_rule(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        warmstart: bool = False,
        verbose: bool = False,
        n_min: int = 0,
        time_limit: int = 300,
        return_opt_flag: bool = False,
    ) -> list[int]:
        """Find a single conjunction with lowest 0-1 error

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            warmstart (bool, optional): If true, an approximate solution will be created first to warmstart the MIO.
                Defaults to False.
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.

        Returns:
            list[int]: List of indices of the literals in the final conjunction
        """
        assert y.shape == (X.shape[0],)
        assert X.dtype == bool and y.dtype == bool

        if warmstart:
            print("No warmstart available, previous attempts were useless")

        # w = np.ones_like(y, dtype=float)
        size1 = np.sum(y)
        if size1 == 0:
            return list(range(X.shape[1]))
        if size1 == y.shape[0]:
            return []
        size0 = y.shape[0] - size1
        # w[y] = 1 / size1
        # w[~y] = 1 / size0

        X0, counts0 = np.unique(X[~y], return_counts=True, axis=0)
        X1, counts1 = np.unique(X[y], return_counts=True, axis=0)
        X = np.concatenate([X0, X1], axis=0)
        w = np.concatenate([counts0 / size0, counts1 / size1], axis=0)
        y = np.zeros_like(w, dtype=bool)
        y[X0.shape[0] :] = True

        # int_model = self._make_int_model(X, y, weights=w)
        int_model = self._make_abs_model(X, y, weights=w, n_min=n_min)
        opt = pyo.SolverFactory("gurobi", solver_io="python")
        opt.options["TimeLimit"] = time_limit
        result = opt.solve(int_model, tee=verbose)
        opt = True
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            # raise ValueError("solver did not find an optimal sollution")
            logger.info("Solver did not find an optimal sollution")
            opt = False
        self.model = int_model

        # print([int_model.error[i].value for i in int_model.all_i])
        vals = [int_model.use_feat[i].value for i in int_model.feat_i]

        rule = [i for i in int_model.feat_i if vals[i] is not None and vals[i] >= 1e-4]
        if return_opt_flag:
            return rule, opt
        return rule

    def find_subgroup(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        verbose: bool = False,
    ) -> np.ndarray[bool]:
        """Find a single conjunction with lowest 0-1 error and returns the y_hat vector of classifications

        Args:
            X (np.ndarray[bool]): Input data (boolean values), shape (n, d)
            y (np.ndarray[bool]): Target (boolean values), shape (n,)
            verbose (bool, optional): If true, solver output is printed to stdout. Defaults to False.

        Returns:
            np.ndarray[int]: List of indices of the literals in the final conjunction
        """
        conjuncts = self.find_rule(X, y, verbose=verbose)
        y_hat = np.ones_like(y, dtype=bool)
        for conj in conjuncts:
            y_hat &= X[:, conj]
        return y_hat
