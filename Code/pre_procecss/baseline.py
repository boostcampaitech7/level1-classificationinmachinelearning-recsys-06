from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class BaseLineData:
    def __init__(self, df: pd.DataFrame):
        self.cols_dict: Dict[str, str] = {
            "ID": "ID",
            "target": "target",
            "_type": "_type",
            "hourly_market-data_coinbase-premium-index_coinbase_premium_gap": "coinbase_premium_gap",
            "hourly_market-data_coinbase-premium-index_coinbase_premium_index": "coinbase_premium_index",
            "hourly_market-data_funding-rates_all_exchange_funding_rates": "funding_rates",
            "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations": "long_liquidations",
            "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd": "long_liquidations_usd",
            "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations": "short_liquidations",
            "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations_usd": "short_liquidations_usd",
            "hourly_market-data_open-interest_all_exchange_all_symbol_open_interest": "open_interest",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio": "buy_sell_ratio",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "buy_volume",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "sell_volume",
            "hourly_network-data_addresses-count_addresses_count_active": "active_count",
            "hourly_network-data_addresses-count_addresses_count_receiver": "receiver_count",
            "hourly_network-data_addresses-count_addresses_count_sender": "sender_count",
        }
        self.df = df
        self.train_df = None
        self.test_df = None
        self._preprocess()

    def _select_columns(self):
        self.df = self.df[self.cols_dict.keys()].rename(self.cols_dict, axis=1)

    def _shift_feature(
        self,
        conti_cols: List[str],
        intervals: List[int],
    ) -> List[pd.Series]:
        """
        연속형 변수의 shift feature 생성
        Args:
            df (pd.DataFrame)
            conti_cols (List[str]): continuous colnames
            intervals (List[int]): shifted intervals
        Return:
            List[pd.Series]
        """
        df_shift_dict = [
            self.df[conti_col].shift(interval).rename(f"{conti_col}_{interval}")
            for conti_col in conti_cols
            for interval in intervals
        ]
        return df_shift_dict

    def _create_new_features(self):
        self.df = self.df.assign(
            liquidation_diff=self.df["long_liquidations"]
            - self.df["short_liquidations"],
            liquidation_usd_diff=self.df["long_liquidations_usd"]
            - self.df["short_liquidations_usd"],
            volume_diff=self.df["buy_volume"] - self.df["sell_volume"],
            liquidation_diffg=np.sign(
                self.df["long_liquidations"] - self.df["short_liquidations"]
            ),
            liquidation_usd_diffg=np.sign(
                self.df["long_liquidations_usd"] - self.df["short_liquidations_usd"]
            ),
            volume_diffg=np.sign(self.df["buy_volume"] - self.df["sell_volume"]),
            buy_sell_volume_ratio=self.df["buy_volume"] / (self.df["sell_volume"] + 1),
        )

    def get_train_test(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self.train_df, self.test_df

    def _preprocess(self):
        # 모델 학습에 필요한 Column 설정
        self._select_columns()
        # eda 에서 파악한 차이와 차이의 음수, 양수 여부를 새로운 피쳐로 생성
        self._create_new_features()
        # category, continuous 열을 따로 할당해둠 - Deprecate(Not Use)
        # category_cols: List[str] = [
        #     "liquidation_diffg",
        #     "liquidation_usd_diffg",
        #     "volume_diffg",
        # ]

        conti_cols: List[str] = [
            _ for _ in self.cols_dict.values() if _ not in ["ID", "target", "_type"]
        ] + [
            "buy_sell_volume_ratio",
            "liquidation_diff",
            "liquidation_usd_diff",
            "volume_diff",
        ]
        # 최대 24시간의 shift 피쳐를 계산
        shift_list = self._shift_feature(
            conti_cols=conti_cols, intervals=[_ for _ in range(1, 24)]
        )

        # concat 하여 df 에 할당
        df = pd.concat([self.df, pd.concat(shift_list, axis=1)], axis=1)

        # 타겟 변수를 제외한 변수를 forwardfill, -999로 결측치 대체
        _target = df["target"]
        df = df.ffill().fillna(-999).assign(target=_target)

        # _type에 따라 train, test 분리
        self.train_df = df.loc[df["_type"] == "train"].drop(columns=["_type"])
        self.test_df = df.loc[df["_type"] == "test"].drop(columns=["_type"])
