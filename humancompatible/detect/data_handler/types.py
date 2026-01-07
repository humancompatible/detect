import numpy as np
import pandas as pd


OneDimData = np.ndarray | pd.Series
"""One-dimensional data: a NumPy array or a pandas Series."""

CategValue = int | str
"""A categorical value: integer-coded or string."""

DataLike = np.ndarray | pd.DataFrame
"""Tabular data: a NumPy array or a pandas DataFrame."""

FeatureID = int | str
"""Feature identifier: column index (int) or column name (str)."""
