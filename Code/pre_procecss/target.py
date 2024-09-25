import pandas as pd

from Code.pre_procecss.interface import PreProcessInterface


class PreProcessor(PreProcessInterface):

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self._preprocess()

    def _preprocess(self):
        df = self.df
        df["open"] = df["target_closed"].shift(1)
        df = df.iloc[1:]
        df["target_ratio"] = ((df["target_closed"] / df["open"]) - 1) * 100
        df["target_ratio"] = df["target_ratio"]  # .shift(-1)

        ## T/F Target Create
        df["target_tf"] = 0
        df.loc[df["target"] >= 2, "target_tf"] = 1
        self.df = df

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        return (
            self.df.loc[self.df["_type"] == "train"],
            self.df.loc[self.df["_type"] == "test"],
        )

    def get_data(self) -> pd.DataFrame:
        return self.df
