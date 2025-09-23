import types
import numpy as np

from humancompatible.detect.methods.l_inf.lp_tools import lin_prog_feas


# =====
# Tests for lin_prog_feas()
# =====
def test_identical_histograms_are_feasible(monkeypatch):
    h = np.array([0.2, 0.3, 0.5], dtype=float)

    # Make sampling deterministic, always pick bin 0
    calls = {"n": 0}
    def _fake_randrange(a, b):  # signature randrange(start, stop)
        calls["n"] += 1
        return 0
    monkeypatch.setattr("humancompatible.detect.methods.l_inf.lp_tools.randrange", _fake_randrange)

    status = lin_prog_feas(h, h.copy(), delta=0.0, num_samples=1.0)
    assert status == 0

def test_detects_violation_when_sampled(monkeypatch):
    # Difference in the first bin exceeds delta -> infeasible if that bin is sampled
    h1 = np.array([0.1, 0.9], dtype=float)
    h2 = np.array([0.9, 0.1], dtype=float)
    delta = 0.2  # |0.1 - 0.9| = 0.8 > 0.2

    monkeypatch.setattr("humancompatible.detect.methods.l_inf.lp_tools.randrange", lambda a, b: 0)

    status = lin_prog_feas(h1, h2, delta=delta, num_samples=1.0)
    assert status != 0

def test_last_bin_not_sampled_doesnt_affect_result(monkeypatch):
    # Only the last bin differs, but the function samples from [0, len-1),
    # so that difference should never be seen => still feasible.
    h1 = np.array([0.5, 0.5], dtype=float)
    h2 = np.array([0.5, 0.9], dtype=float)
    delta = 1e-6

    monkeypatch.setattr("humancompatible.detect.methods.l_inf.lp_tools.randrange", lambda a, b: 0)

    status = lin_prog_feas(h1, h2, delta=delta, num_samples=1.0)
    assert status == 0 

def test_builds_expected_number_of_constraints(monkeypatch):
    # Verify the number of sampled rows translates into A_ub/b_ub shapes: (2k, 1)
    h1 = np.array([0.2, 0.3, 0.5, 0.0], dtype=float)
    h2 = np.array([0.2, 0.3, 0.5, 0.0], dtype=float)
    n_bins = h1.shape[0]
    frac = 0.75
    k = int(frac * (n_bins - 1))

    # Stub linprog to avoid solver dependency and to introspect inputs
    captured = {}
    def _fake_linprog(c, A_ub, b_ub, bounds):
        captured["c"] = c
        captured["A_ub_shape"] = np.shape(A_ub)
        captured["b_ub_shape"] = np.shape(b_ub)
        captured["bounds"] = bounds
        # Return an object with .status == 0
        r = types.SimpleNamespace()
        r.status = 0
        return r

    monkeypatch.setattr("humancompatible.detect.methods.l_inf.lp_tools.randrange", lambda a, b: 0)
    monkeypatch.setattr("humancompatible.detect.methods.l_inf.lp_tools.optimize.linprog", _fake_linprog)

    status = lin_prog_feas(h1, h2, delta=0.1, num_samples=frac)
    assert status == 0
    assert captured["A_ub_shape"] == (2 * k, 1)
    assert captured["b_ub_shape"] == (2 * k, 1)
    assert captured["bounds"] == [(1, 1)]
