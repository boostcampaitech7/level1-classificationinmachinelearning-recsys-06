import pandas as pd

from Code.pre_procecss.interface import PreProcessInterface


class PreProcessor(PreProcessInterface):

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self._preprocess()
        pass

    def _preprocess(self):
        self.df["sell_buy_01"] = 0
        self.df["buy_r_v"] = (
                self.df["all_exchange_taker_buy_ratio"]
                * self.df["all_exchange_taker_buy_volume"]
        )
        self.df["sell_r_v"] = (
                self.df["all_exchange_taker_sell_ratio"]
                * self.df["all_exchange_taker_sell_volume"]
        )
        self.df["sell_buy_r_v_diff"] = self.df["sell_r_v"] - self.df["buy_r_v"]
        self.df.loc[
            self.df["all_exchange_taker_buy_ratio"]
            > self.df["all_exchange_taker_sell_ratio"],
            "sell_buy_01",
        ] = 1

    def get_train_test(self):
        return (
            self.df.loc[self.df["_type"] == "train"],
            self.df.loc[self.df["_type"] == "test"],
        )

    def get_data(self) -> pd.DataFrame:
        return self.df
