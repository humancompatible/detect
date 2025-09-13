import numpy as np
import pytest

import humancompatible.detect.helpers.utils as utils

def test_evaluate_subgroup_discrepancy_is_abs():
    subgroup = np.array([True, False, False, False])
    y = np.array([True, True,  False, False])
    signed = utils.signed_subgroup_discrepancy(subgroup, y)
    absval = utils.evaluate_subgroup_discrepancy(subgroup, y)
    assert absval == pytest.approx(abs(signed))
