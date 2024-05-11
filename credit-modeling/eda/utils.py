import numpy as np
import pandas as pd
import polars as pl


def raise_for_nan(data):
    has_nan = False

    has_nan = isinstance(data, np.ndarray) and np.isnan(data).any()
    has_nan = isinstance(data, pd.Series) and (data.isnull().any() or data.isna().any())
    has_nan = isinstance(data, pl.Series) and (
        data.is_nan().any() or data.is_null().any()
    )

    if has_nan:
        raise ValueError("NaN/Null values found in data.")
    # raise TypeError(f"Unsupported datatype {type(data)}")
