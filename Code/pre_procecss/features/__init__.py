from typing import Dict

import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_scaled_dict_and_transform_scaled(
        df: pd.DataFrame,
        columns: [str] | str,
        col_name_prefix: str = "scaled",
        scaled_func=StandardScaler,
) -> (pd.DataFrame, Dict):
    def set_sc(col, sc_func) -> (pd.DataFrame, any):
        scaler = sc_func()
        if col_name_prefix != "":
            colname = f"{col_name_prefix}_{col}"
        else:
            colname = col
        df.loc[:, colname] = scaler.fit_transform(df[col].to_numpy().reshape(-1, 1))
        return df, scaler

    scaled_dict = {}
    if type(columns) == str:
        df, scaled_dict[columns] = set_sc(columns, scaled_func)
        return df, scaled_dict

    for c in columns:
        df, scaled_dict[c] = set_sc(c, scaled_func)

    return df, scaled_dict
