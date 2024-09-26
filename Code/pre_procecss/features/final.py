from typing import Dict

from keras.src.legacy.backend import moving_average_update

from Code.pre_procecss.interface import PreProcessInterface

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class PreProcessor(PreProcessInterface):
    """
    해당 PreProcessor는 김건율님의 Feature 엔지니어링 코드 바탕으로 만들어졌습니다.
    Feature 선정 방법은 Readme.md 혹은 Notion을 참고 부탁드립니다.
    """

    def __init__(self, df: pd.DataFrame):
        super(PreProcessor, self).__init__(df)
        pass

    def _select_features(self):
        select_col = [
            "ID",
            "_type",
            "target",
            "hourly_network-data_hashrate_hashrate",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio",
            "hourly_market-data_open-interest_all_exchange_all_symbol_open_interest",
            "hourly_market-data_funding-rates_all_exchange_funding_rates",
            "hourly_market-data_coinbase-premium-index_coinbase_premium_index",
            "hourly_network-data_utxo-count_utxo_count",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume",
            "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations",
            "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations",
            "hourly_network-data_fees_fees_block_mean_usd",
            "hourly_network-data_fees_fees_reward_percent",
            "hourly_network-data_transactions-count_transactions_count_total",
        ]
        self.df = self.df[select_col]

    def _rename_column(self):
        cols_dict: Dict[str, str] = {
            "ID": "ID",
            "_type": "_type",
            "target": "target",
            "spot_closed_difference": "difference",
            "spot_closed_percent": "percent",
            "hourly_network-data_hashrate_hashrate": "hashrate",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_sell_ratio": "taker_buy_sell_ratio",
            "hourly_market-data_open-interest_all_exchange_all_symbol_open_interest": "open_interest",
            "hourly_market-data_funding-rates_all_exchange_funding_rates": "funding_rates",
            "hourly_market-data_coinbase-premium-index_coinbase_premium_index": "coinbase_premium_index",
            "hourly_network-data_utxo-count_utxo_count": "utxo_count",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume": "taker_sell_volume",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume": "taker_buy_volume",
            "hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations": "long_liquidations",
            "hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations": "short_liquidations",
            "hourly_network-data_fees_fees_block_mean_usd": "fees_block_mean_usd",
            "hourly_network-data_fees_fees_reward_percent": "fees_reward_percent",
            "hourly_network-data_transactions-count_transactions_count_total": "transactions_count_total",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_ratio": "sell_ratio",
            "hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_ratio": "buy_ratio",
            "hourly_network-data_tokens-transferred_tokens_transferred_total": "tokens_transferred_total",
            "hourly_network-data_block-bytes_block_bytes": "block_bytes",
            "hourly_network-data_block-count_block_count": "block_count",
        }
        self.df = self.df[list(cols_dict.keys())].rename(cols_dict, axis=1)

    def _create_new_features(self):
        self.df = self.df.assign(
            ## LJJ Features
            average_transaction_value=self.df[
                                          "tokens_transferred_total"
                                      ]  # 평균 트랜잭션 가치
                                      / self.df["transactions_count_total"],
            network_load=self.df["block_bytes"]
                         / self.df["block_count"],  # 네워크 부하 지표
            market_pressure=self.df["buy_ratio"]
                            / self.df["sell_ratio"],  # 시장 압력 지표
            liquidation_risk=(  # 청산 위험 지표
                                     self.df["long_liquidations"] + self.df["short_liquidations"]
                             )
                             / self.df["open_interest"],
            ## KKY Features
            ID=pd.to_datetime(self.df["ID"]),
            total_liquidations=(
                    self.df["long_liquidations"] + self.df["short_liquidations"]
            ),
            liquidation_diff=(
                    self.df["long_liquidations"] - self.df["short_liquidations"]
            ),
            total_taker_volume=(
                    self.df["taker_sell_volume"] + self.df["taker_buy_volume"]
            ),
            estimated_block_reward_usd=(
                    self.df["fees_block_mean_usd"] / self.df["fees_reward_percent"]
            ),
            open_interst_diff=self.df["open_interest"].diff(),
            moving_avg_open_interest=self.df["open_interest"]
            .rolling(window=24, min_periods=1)
            .mean(),
        )

    def _transform_features(self):
        col_log_dict = {
            # Column-name: Log 변환 여부
            "hashrate": True,
            "open_interest": True,
            "coinbase_premium_index": True,
            "funding_rates": True,
            "estimated_block_reward_usd": True,
            "total_liquidations": True,
            "total_taker_volume": True,
            "liquidation_diff": False,
            "utxo_count": False,
            "transactions_count_total": False,
        }
        # 훈련 및 테스트 데이터프레임 분리
        train_df = self.df.loc[self.df["_type"] == "train"].copy()
        test_df = self.df.loc[self.df["_type"] == "test"].copy()
        # 스케일러 정의
        from Code.pre_procecss.features import get_scaled_dict_and_transform_scaled

        for k, v in col_log_dict:
            feature = train_df[k].values.reshape(-1, 1)
            name_prefix = "scaled"
            if v is True:
                feature = np.log1p(feature)
                name_prefix = "scaled_log"
            feature, scaler = get_scaled_dict_and_transform_scaled(
                feature, k, name_prefix
            )
            train_df[k] = feature
            test_df[k] = scaler(test_df[k].values.reshape(-1, 1))

        # moving_avg_scaled_log_total_volume 표준화
        train_df["moving_avg_scaled_log_total_volume"] = (
            train_df["scaled_log_total_taker_volume"]
            .rolling(window=24, min_periods=1)
            .mean()
        )
        test_df["moving_avg_scaled_log_total_volume"] = (
            test_df["scaled_log_total_taker_volume"]
            .rolling(window=24, min_periods=1)
            .mean()
        )
        self.df = pd.concat([train_df, test_df])

    def _preprocess(self):
        self._rename_column()
        # New Features
        self._create_new_features()

        # 최종 데이터프레임 통합
        selected_columns = [
            "ID",
            "_type",
            "target",
            "scaled_log_hashrate",
            "scaled_log_open_interest",
            "scaled_log_coinbase_premium_index",
            "scaled_funding_rates",
            "scaled_estimated_block_reward",
            "scaled_liquidation_diff",
            "scaled_log_total_liquidation",
            "scaled_log_total_taker_volume",
            "scaled_utxo_count",
            "scaled_total_transactions_count",
            "taker_buy_sell_ratio",
            "moving_avg_scaled_log_total_volume",
            "open_interest_diff",
            "average_transaction_value",
            "network_load",
            "market_pressure",
            "liquidation_risk",
        ]
        self.df = self.df[selected_columns]

    def get_data(self) -> pd.DataFrame:
        return self.df

    def get_train_test(self) -> (pd.DataFrame, pd.DataFrame):
        df = self.df
        return df.loc[df["_type"] == "train"], df.loc[df["_type"] == "test"]
