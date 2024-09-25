import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler

from Code.pre_procecss.interface import PreProcessInterface

class PreProcessor(PreProcessInterface):


    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self._preprocess()
        pass

    def _preprocess(self):
        df = self.df

        cols_dict: Dict[str, str] = {
            'ID': 'ID',
            '_type': '_type',
            'target': 'target',
            'hourly_network-data_hashrate_hashrate' : 'hashrate',
            'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume' : 'taker_sell_volume', 
            'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume' : 'taker_buy_volume', 
            'hourly_market-data_open-interest_all_exchange_all_symbol_open_interest' : 'open_interest', 
            'hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations' : 'long_liquidations',
            'hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations_usd' : 'long_liquidations_usd', 
            'hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations' : 'short_liquidations', 
            'hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations_usd' : 'short_liquidations_usd'
        }

        df = df.rename(columns=cols_dict)

        train_df = df.loc[df['_type'] == 'train']
        test_df = df.loc[df['_type'] == 'test']
        
        train_scaler = StandardScaler()
        test_scaler = StandardScaler()

        # 각 특성에 대해 로그 변환 및 표준화 적용
        features = [
            'hourly_network-data_hashrate_hashrate',
            'hourly_market-data_open-interest_all_exchange_open_interest',
            'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume',
            'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume'
        ]

        
        for feature in features:
            # train 데이터 처리
            train_log_feature = np.log1p(train_df[feature])
            train_df[f'scaled_log_{feature}'] = train_scaler.fit_transform(train_log_feature.values.reshape(-1, 1))
            
            # test 데이터 처리
            test_log_feature = np.log1p(test_df[feature])
            test_df[f'scaled_log_{feature}'] = test_scaler.fit_transform(test_log_feature.values.reshape(-1, 1))

        # total volume 생성과 처리
        for df_type in [train_df, test_df]:
            total_volume = df_type['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume'] + \
                           df_type['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume']
            log_total_volume = np.log1p(total_volume)
            
            if df_type is train_df:
                df_type['scaled_log_total_volume'] = train_scaler.fit_transform(log_total_volume.values.reshape(-1, 1))
            else:
                df_type['scaled_log_total_volume'] = test_scaler.fit_transform(log_total_volume.values.reshape(-1, 1))
            
            df_type['moving_avg_scaled_log_total_volume'] = df_type['scaled_log_total_volume'].rolling(window=24, min_periods=1).mean()

        # liquidation feature 생성과 처리
        for df_type in [train_df, test_df]:
            liquidation_diff = df_type['hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations'] - \
                               df_type['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']
            total_liquidation = df_type['hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations'] + \
                                df_type['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']
            log_total_liquidation = np.log1p(total_liquidation)
            
            if df_type is train_df:
                df_type['scaled_liquidation_diff'] = train_scaler.fit_transform(liquidation_diff.values.reshape(-1, 1))
                df_type['scaled_log_total_liquidation'] = train_scaler.fit_transform(log_total_liquidation.values.reshape(-1, 1))
            else:
                df_type['scaled_liquidation_diff'] = test_scaler.fit_transform(liquidation_diff.values.reshape(-1, 1))
                df_type['scaled_log_total_liquidation'] = test_scaler.fit_transform(log_total_liquidation.values.reshape(-1, 1))

        # train_df와 test_df를 다시 합치기
        self.df = pd.concat([train_df, test_df])
    
    def get_train_test(self):
        return (
        self.df.loc[self.df["_type"] == "train"],
        self.df.loc[self.df["_type"] == "test"],
        )

    def get_data(self) -> pd.DataFrame:
        return self.df



