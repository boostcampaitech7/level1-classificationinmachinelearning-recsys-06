import pandas as pd

from Code.pre_procecss.interface import PreProcessInterface


class PreProcessor(PreProcessInterface):
    """
       새로 생성된 feature
       해당 Features은 Column_rename PreProcessor 를 호출 후에 가능.
    - sell_buy_01 : 판매 비율이 높으면 0 / 구매비율이 높으면 1인 Feature
    - sell_r_v / buy_r_v : sell/buy ratio * sell/buy volume
    - sell_buy_r_v_diff : sell_r_v - buy_r_v
    - s
    """

    def __init__(self, df: pd.DataFrame):
        super(PreProcessor, self).__init__(df)

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
