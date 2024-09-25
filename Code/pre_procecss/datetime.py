import pandas as pd

from Code.pre_procecss.interface import PreProcessInterface


class PreProcessor(PreProcessInterface):

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self._preprocess()

    def _preprocess(self):
        df = self.df
        df["ID"] = pd.to_datetime(df["ID"])
        df["hour"] = df["ID"].map(lambda x: x.hour)
        df["weekday"] = df["ID"].map(lambda x: x.day_of_week)
        self.df = df

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        return (
            self.df.loc[self.df["_type"] == "train"],
            self.df.loc[self.df["_type"] == "test"],
        )

    def get_data(self) -> pd.DataFrame:
        return self.df
