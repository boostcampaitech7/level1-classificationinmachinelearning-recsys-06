import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler

from Code.pre_procecss.interface import PreProcessInterface


class PreProcessor(PreProcessInterface):
    """
    새로 생성된 feature
    - scaled_log_hashrate : hashrate에 대해 로그변환 및 표준화
    - scaled_log_open_interest : open_interest에 대해 로그변환 및 표준화
    - scaled_log_taker_buy_volume : taker_buy_volume에 대해 로그변환 및 표준화
    - scaled_log_taker_sell_volume : taker_sell_volume에 대해 로그변환 및 표준화
    - scaled_log_total_volume : taker_buy_volume와 taker_sell_volume의 합에 대해 로그변환 및 표준화
    - moving_avg_scaled_log_total_volume : scaled_log_total_volume에 대해 24시간 이동평균 적용
    - scaled_liquidation_diff : long_liquidations과 short_liquidations의 차에 대해 표준화
    - scaled_log_total_liquidation : long_liquidations와 short_liquidations의 합에 대해 로그변환 및 표준화

    기존 raw.csv 파일의 feature를 받아서 새로운 feature를 생성해 이를 더해서 반환(기존 피처 유지)
    """

    def __init__(self, df: pd.DataFrame):
        super(PreProcessor, self).__init__(df)

    @staticmethod
    def get_scaled_dict_and_transform_scaled(df: pd.DataFrame, columns: [str] | str, col_name_prefix: str = "scaled",
                                             scaled_func=StandardScaler) -> (pd.DataFrame, Dict):
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

    def _preprocess(self):
        df = self.df
        total_volume = df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume'] + \
                       df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume']
        liquidation_diff = df[
                               'hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations'] - \
                           df['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']
        total_liquidation = df[
                                'hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations'] + \
                            df[
                                'hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']

        # 각 특성에 대해 로그 변환 및 표준화 적용
        features: Dict[str, str | pd.Series] = {
            'hourly_network-data_hashrate_hashrate': 'hashrate',
            'hourly_market-data_open-interest_all_exchange_all_symbol_open_interest': 'open_interest',
            'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume': 'taker_buy_volume',
            'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume': 'taker_sell_volume',
            'total_liquidation': total_liquidation,
            'liquidation_diff': liquidation_diff,
            'total_volume': total_volume,
        }
        self.get_scaled_dict_and_transform_scaled(df, features.keys())

        for k, v in features.items():
            # train 데이터 처리
            scaler = StandardScaler()
            if type(v) == str:
                feature = np.log1p(df[k])
            elif v == "liquidation_diff":
                feature = v
            else:
                feature = np.log1p(v)
            df.loc[:, f'scaled_log_{v}'] = scaler.fit_transform(feature.values.reshape(-1, 1))

        df.loc[:, 'moving_avg_scaled_log_total_volume'] = df['scaled_log_total_volume'].rolling(window=24,
                                                                                                min_periods=1).mean()
        self.df = df

    def get_train_test(self):
        return (
            self.df.loc[self.df["_type"] == "train"],
            self.df.loc[self.df["_type"] == "test"],
        )

    def get_data(self) -> pd.DataFrame:
        return self.df
