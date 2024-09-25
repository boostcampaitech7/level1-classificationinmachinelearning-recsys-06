import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from Code.pre_procecss.interface import PreProcessInterface

class PreProcessor(PreProcessInterface):


    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self._preprocess()
        pass

    def _preprocess(self):
        df = self.df
        # hashrate 로그 변환 후 표준화
        df['scaled_log_hashrate'] = StandardScaler().fit_transform(np.log1p(df['hourly_network-data_hashrate_hashrate'].replace(0, np.nan)).dropna())
        # open_interest 로그 변환 후 표준화
        df['scaled_log_open_interest'] = StandardScaler().fit_transform(np.log1p(df['hourly_market-data_open-interest_all_exchange_open_interest'].replace(0, np.nan)).dropna())
        # taker_volume 로그 변환 후 표준화
        df['scaled_log_taker_buy_volume'] = StandardScaler().fit_transform(np.log1p(df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume'].replace(0, np.nan)).dropna())
        df['scaled_log_taker_sell_volume'] = StandardScaler().fit_transform(np.log1p(df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume'].replace(0, np.nan)).dropna())

        # total volume 로그 변환 후 표준화 / total volume의 이동 평균(24시간)
        total_volume = df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume'] + df['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume']
        df['scaled_log_total_volume'] = StandardScaler().fit_transform(np.log1p(total_volume.replace(0, np.nan)).dropna())
        df['moving_avg_scaled_log_taker_total_volume'] = df['scaled_log_taker_total_volume'].rolling(window=24, min_periods=1).mean()

        # liquidation diff / total liquidation 생성 후 로그 변환과 표준화
        liquidaion_diff = df['hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations'] - df['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']
        df['scaled_liquidation_diff'] = StandardScaler().fit_transform(liquidaion_diff)
        total_liquidation = df['hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations'] + df['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']
        df['scaled_log_total_liquidation'] = StandardScaler().fit_transform(np.log1p(total_liquidation.replace(0, np.nan)).dropna())
        self.df = df
    
    def get_train_test(self):
        return (
        self.df.loc[self.df["_type"] == "train"],
        self.df.loc[self.df["_type"] == "test"],
        )

    def get_data(self) -> pd.DataFrame:
        return self.df



