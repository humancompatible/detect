# from __future__ import annotations

import numpy as np
import pandas as pd

# import sys


# if sys.version_info[1] < 10:
#     from typing import Union

#     OneDimData = Union[np.ndarray, pd.Series]
#     CategValue = Union[int, str]
#     DataLike = Union[np.ndarray, pd.DataFrame]
#     FeatureID = Union[int, str]
# else:
OneDimData = np.ndarray | pd.Series
CategValue = int | str
DataLike = np.ndarray | pd.DataFrame
FeatureID = int | str
