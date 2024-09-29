import pandas as pd

from Code.pre_procecss.interface import PreProcessInterface


class PreProcessor(PreProcessInterface):

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self._preprocess()

    def _preprocess(self):
        # 불 필요한 Column-name Split을 통한 삭제
        cols = self.df.columns
        new_cols = []
        for col in cols:
            if col in ["ID", "target", "_type"]:
                new_cols.append(col)
                continue
            col = f"{col}".replace("hourly_", "").split("_", maxsplit=2)[2]
            new_cols.append(col)
        self.df.columns = new_cols

        # target 관련된 종가/거래량 column-name 변경
        rename_dict = {
            "all_exchange_spot_btc_usd_close": "target_closed",
            "all_exchange_spot_btc_usd_volume": "target_volume",
        }
        self.df.rename(columns=rename_dict, inplace=True)

        return

    def get_train_test(self):
        return (
            self.df.loc[self.df["_type"] == "train"],
            self.df.loc[self.df["_type"] == "test"],
        )

    def get_data(self) -> pd.DataFrame:
        return self.df
