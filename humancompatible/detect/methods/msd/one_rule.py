import logging
from typing import List, Tuple, Dict

import numpy as np
import pyomo.environ as pyo

logger = logging.getLogger(__name__)


class OneRule:
    """Implementation of a MIO formulation for finding an optimal conjunction.

    This class implements a Mixed-Integer Optimization (MIO) formulation to
    discover an optimal conjunction (a logical AND of features) that maximizes
    the absolute difference in target outcomes between the subgroup defined by
    this conjunction and its complement. The formulation is inspired by the
    1Rule method from the paper "Learning Optimal and Fair Classifiers" by
    Malioutov and Varshney (http://proceedings.mlr.press/v28/malioutov13.pdf).
    """

    def __init__(self) -> None:
        """
        Initializes the OneRule solver.
        """
        # Stores the Pyomo model after solving
        self.model: pyo.ConcreteModel | None = None

    def _make_abs_model(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        weights: np.ndarray[float],
        n_min: int,
        feat_init: Dict[int, int] | None = None,
    ) -> pyo.ConcreteModel:
        """
        Creates the Mixed-Integer Optimization (MIO) formulation to find an optimal conjunction.

        This private method constructs the Pyomo model for the MIO problem.
        The objective is to maximize the absolute difference in weighted sums
        of positive outcomes between the subgroup and its complement, subject
        to constraints that define the subgroup based on selected features
        and a minimum support.

        Args:
            X (np.ndarray[bool]): The input feature matrix, where each row is an
                instance and each column is a boolean feature.
                Shape `(n_instances, n_features)`.
            y (np.ndarray[bool]): The binary target labels for each instance.
                `True` for positive outcomes, `False` for negative.
                Shape `(n_instances,)`.
            weights (np.ndarray[float]): Weights for each instance, used in the
                objective function to calculate weighted means.
                Typically, these are normalized counts.
                Shape `(n_instances,)`.
            n_min (int): Minimum subgroup support (number of instances) required
                for a valid subgroup.
            feat_init (Dict[int, int] | None, default None): Initialization for the `use_feat`
                binary variables. A dictionary where keys are feature indices and
                values are `0` (feature not used) or `1` (feature used) in the
                conjunction. Defaults to an empty dictionary, allowing the solver
                to determine initial values.

        Returns:
            pyo.ConcreteModel: The constructed Pyomo MIO model instance.

        Notes:
            - The model uses `pyomo.Binary` variables for `use_feat` to select features
              for the conjunction.
            - `ingroup` variables determine if an instance belongs to the current subgroup.
            - `force_0` and `force_1` constraints ensure `ingroup[i]` is 1 if and only if
              all `use_feat[j]` that are 1 also have `Xint[i, j]` as 1.
            - `minimum` constraint enforces the `n_min` support.
            - `o` and `b` variables, along with `abs_obj_u1` and `abs_obj_u2` constraints,
              linearize the absolute value in the objective function.
        """
        if feat_init is None:
            feat_init = {}
        
        n, d = X.shape
        # Convert boolean X to integer for Pyomo compatibility (0 or 1)
        Xint = np.zeros_like(X, dtype=int)
        Xint[X] = 1

        model = pyo.ConcreteModel()
        model.all_i = pyo.Set(initialize=np.arange(n))
        model.feat_i = pyo.Set(initialize=np.arange(d))
        model.pos_i = pyo.Set(initialize=np.where(y)[0])  # Indices of positive outcomes
        model.neg_i = pyo.Set(initialize=np.where(~y)[0])  # Indices of negative ones

        # Decision variables
        # `use_feat[j]` is 1 if feature `j` is part of the conjunction, 0 otherwise.
        model.use_feat = pyo.Var(model.feat_i, domain=pyo.Binary, initialize=feat_init)
        # `ingroup[i]` is 1 if instance `i` is in the subgroup, 0 otherwise.
        model.ingroup = pyo.Var(model.all_i, domain=pyo.NonNegativeReals, bounds=(0, 1))

        # Constraints to define subgroup membership:

        # If a feature `j` is used (`use_feat[j] == 1`) AND instance `i` does NOT
        # have that feature (`Xint[i, j] == 0`), then `ingroup[i]` must be 0.
        model.force_0 = pyo.Constraint(
            model.all_i,
            model.feat_i,
            rule=lambda m, i, j: (
                m.ingroup[i] <= 1 - (m.use_feat[j] - Xint[i, j] * m.use_feat[j])
            ),
        )
        # If an instance `i` has all features selected in the conjunction,
        # then `ingroup[i]` must be 1.
        model.force_1 = pyo.Constraint(
            model.all_i,
            rule=lambda m, i: (
                m.ingroup[i]
                >= 1 - sum(m.use_feat[j] - Xint[i, j] * m.use_feat[j] for j in m.feat_i)
            ),
        )

        # Minimum subgroup support constraint
        # The sum of `ingroup` variables (total members in subgroup) must be at least `n_min`.
        model.minimum = pyo.Constraint(
            expr=sum(model.ingroup[i] for i in model.all_i) >= n_min
        )

        # Variables and constraints for linearizing the absolute value in the objective
        # `o` represents the absolute difference we want to maximize.
        model.o = pyo.Var(domain=pyo.NonNegativeReals)
        # `b` is a binary variable used for linearization.
        model.b = pyo.Var(domain=pyo.Binary)

        # Calculate weighted sums for positive and negative outcomes within the subgroup
        term1 = sum(model.ingroup[i] * weights[i] for i in model.pos_i)
        term2 = sum(model.ingroup[i] * weights[i] for i in model.neg_i)

        # Linearization of `abs(term1 - term2)`:
        # `o <= (term1 - term2) + M * b`
        # `o <= (term2 - term1) + M * (1 - b)`
        # (where M is a sufficiently large number, here implicitly 2, because weights sum to 1)
        model.abs_obj_u1 = pyo.Constraint(expr=model.o <= term1 - term2 + 2 * model.b)
        model.abs_obj_u2 = pyo.Constraint(
            expr=model.o <= term2 - term1 + 2 * (1 - model.b)
        )
        # Objective function: maximize the absolute difference `o`
        model.obj = pyo.Objective(
            expr=model.o,
            sense=pyo.maximize,
        )

        return model

    def _make_solver(self, solver_name: str, verbose: int = 2, time_limit: int = 300):
        # Solver setup
        if solver_name == "gurobi":
            solver = pyo.SolverFactory(solver_name, solver_io="python")
        else:
            solver = pyo.SolverFactory(solver_name)

        opts = {}

        # Set time limit and log outputs parameter for the solver
        if "cplex" in solver_name:
            opts["mip display"] = 4 if verbose == 2 else 0
            opts["simplex display"] = 2 if verbose == 2 else 0
            opts["bar display"] = 1 if verbose == 2 else 0
            opts["timelimit"] = time_limit
        elif "glpk" in solver_name:
            opts["msg_lev"] = "GLP_MSG_ALL" if verbose == 2 else "GLP_MSG_OFF"
            opts["tmlim"] = time_limit
        elif "xpress" in solver_name:
            opts["outputlog"] = 1 if verbose == 2 else 0
            opts["maxtime"] = time_limit
            opts["soltimelimit"] = time_limit
        elif "highs" in solver_name:
            opts["log_to_console"] = True if verbose == 2 else False
            opts["output_flag"] = 1 if verbose == 2 else 0
            opts["time_limit"] = time_limit
        elif "gurobi" in solver_name:
            opts["OutputFlag"] = 1 if verbose == 2 else 0
            opts["TimeLimit"] = time_limit
        else:
            if verbose >= 1: logger.warning(
                f'Time limit not set! Not implemented for the selected solver "{solver_name}".'
            )

        solver.options.update({k: v for k, v in opts.items() if v is not None})
        return solver

    def find_rule(
        self,
        X: np.ndarray[bool],
        y: np.ndarray[bool],
        n_min: int = 0,
        time_limit: int = 300,
        solver_name: str = "appsi_highs",
        verbose: int = 1,
    ) -> Tuple[List[int] | None, bool]:
        """
        Finds a single conjunction (rule) that maximizes the absolute difference
        in target outcomes between the subgroup it defines and its complement.

        This method prepares the data (by creating unique rows and assigning weights),
        builds the MIO model using `_make_abs_model`, and then solves it using whichever 
        solver you specify in `solver_name`.

        Args:
            X (np.ndarray[bool]): Input data matrix of boolean features,
                shape `(n_instances, n_features)`.
            y (np.ndarray[bool]): Target labels (binary), shape `(n_instances,)`.
            n_min (int, default 0): Minimum subgroup support (number of rows)
                required for a valid subgroup.
            time_limit (int, default 300): Time budget for the solver (in seconds).
                Note that only some solvers support this option.
            solver_name (str, default "appsi_highs"): Method for solving the MIO formulation. 
                Can be chosen among:
                
                - "appsi_highs"
                - "gurobi"
                - "cplex"
                - "glpk"
                - "xpress"
                - Other solvers, see Pyomo documentation 
                
                (Note that only the 5 solvers above support the graceful `time_limit`)
            verbose (int, default 1): Verbosity level. 0 = silent, 1 = logger output only,
                2 = all detailed logs (including solver output).

        Returns:
            Tuple[List[int] | None, bool]: A tuple of a list of integer indices representing 
                the features (literals) that form the optimal conjunction. 
                These indices correspond to the columns in the input `X` that define the subgroup.
                If the solver fails to find any feasible solution within the time budget,
                `None` is returned instead.
                The boolean flag is `True` if the returned solution is globally optimal.

        Raises:
            AssertionError: If `y`'s shape is not (X.shape[0],) or if `X` or `y`
                are not of boolean dtype.
            ValueError: If the solver terminates with condition other than timeout, optimality or infeasibility. 
            Exception: Any exceptions raised by Pyomo or solver during model
                creation or solving.

        Notes:
            - The input `X` and `y` are first processed to get unique rows and
              assign weights based on their original counts and class proportions.
              This helps in handling duplicate rows efficiently.
            - Requires a compatible MIP solver; e.g. Gurobi, HiGHS solver to be installed and configured for Pyomo.
            - The `rule` returned contains indices of the *original* features
              (columns of `X`) that define the conjunction.
        """
        assert y.shape == (X.shape[0],)
        assert X.dtype == bool and y.dtype == bool
        assert n_min <= min(sum(y), sum(~y))

        # Handle edge cases where target is all positive or all negative
        size1 = np.sum(y)
        if size1 == 0:
            if verbose >= 1: logger.info(
                "Target 'y' contains no positive outcomes. Returning all features as rule."
            )
            return list(range(X.shape[1]))
        if size1 == y.shape[0]:
            if verbose >= 1: logger.info(
                "Target 'y' contains only positive outcomes. Returning empty rule."
            )
            return []
        size0 = y.shape[0] - size1

        # Aggregate identical rows and calculate weights
        # This step is crucial for efficiency if X contains many duplicate rows.
        # X0: unique rows where y is False, counts0: their frequencies
        X0, counts0 = np.unique(X[~y], return_counts=True, axis=0)
        # X1: unique rows where y is True, counts1: their frequencies
        X1, counts1 = np.unique(X[y], return_counts=True, axis=0)

        # Concatenate unique rows and calculate normalized weights
        X_unique = np.concatenate([X0, X1], axis=0)
        # Weights are normalized by the total count of their respective class
        weights_unique = np.concatenate([counts0 / size0, counts1 / size1], axis=0)
        # Recreate 'y' for the unique rows (False for original negatives, True for original positives)
        y_unique = np.zeros_like(weights_unique, dtype=bool)
        y_unique[X0.shape[0] :] = True  # Mark rows from X1 as True

        # Create and solve the MIO model
        int_model = self._make_abs_model(
            X_unique, y_unique, weights=weights_unique, n_min=n_min
        )

        # Solve the model
        solver = self._make_solver(solver_name, verbose=verbose, time_limit=time_limit)
        result = solver.solve(int_model, load_solutions=False, tee=(verbose == 2))

        is_optimal = True
        if result.solver.termination_condition != pyo.TerminationCondition.optimal:
            is_optimal = False
            if verbose >= 1: logger.info("Solver did not prove optimality of the solution.")
            if (
                result.solver.termination_condition
                == pyo.TerminationCondition.maxTimeLimit
            ):
                if verbose >= 1: logger.info("Timed out.")
            elif result.solver.termination_condition in [
                pyo.TerminationCondition.infeasible,
                pyo.TerminationCondition.infeasibleOrUnbounded,  # problem is always bounded by 0
            ]:
                if verbose >= 1: logger.info("Infeasible formulation, something went wrong.")
            else:
                raise ValueError(
                    f"Unexpected termination condition: {result.solver.termination_condition}."
                )

        try:
            int_model.solutions.load_from(result)
        except ValueError:
            if verbose >= 1: logger.info("No solution found. Try increasing `time_limit`.")
            return None, False

        self.model = int_model  # Store the solved model instance

        # Extract the chosen features from the model's solution
        # `use_feat[i].value` will be approximately 1 for selected features, 0 otherwise.
        vals = [int_model.use_feat[i].value for i in int_model.feat_i]
        # Collect indices of features that are selected (value close to 1)
        rule = [i for i in int_model.feat_i if vals[i] is not None and vals[i] >= 1e-4]

        return rule, is_optimal
