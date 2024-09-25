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

    def _preprocess(self):
        df = self.df

        train_df = df.loc[df['_type'] == 'train'].copy()
        test_df = df.loc[df['_type'] == 'test'].copy()
        
        train_scaler = StandardScaler()
        test_scaler = StandardScaler()

        # 각 특성에 대해 로그 변환 및 표준화 적용
        features: Dict[str, str] = {
            'hourly_network-data_hashrate_hashrate' : 'hashrate',
            'hourly_market-data_open-interest_all_exchange_all_symbol_open_interest' :'open_interest',
            'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume' : 'taker_buy_volume',
            'hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume' : 'taker_sell_volume'
        }

        for k, v in features.items():
            # train 데이터 처리
            train_log_feature = np.log1p(train_df[k])
            train_df.loc[:, f'scaled_log_{v}'] = train_scaler.fit_transform(train_log_feature.values.reshape(-1, 1))
            
            # test 데이터 처리
            test_log_feature = np.log1p(test_df[k])
            test_df.loc[:, f'scaled_log_{v}'] = test_scaler.fit_transform(test_log_feature.values.reshape(-1, 1))

        # total volume 생성과 처리
        for df_type, scaler in zip([train_df, test_df], [train_scaler, test_scaler]):
            total_volume = df_type['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_buy_volume'] + \
                        df_type['hourly_market-data_taker-buy-sell-stats_all_exchange_taker_sell_volume']
            log_total_volume = np.log1p(total_volume)
            
            df_type.loc[:, 'scaled_log_total_volume'] = scaler.fit_transform(log_total_volume.values.reshape(-1, 1))
            df_type.loc[:, 'moving_avg_scaled_log_total_volume'] = df_type['scaled_log_total_volume'].rolling(window=24, min_periods=1).mean()

        # liquidation feature 생성과 처리
        for df_type, scaler in zip([train_df, test_df], [train_scaler, test_scaler]):
            liquidation_diff = df_type['hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations'] - \
                            df_type['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']
            total_liquidation = df_type['hourly_market-data_liquidations_all_exchange_all_symbol_long_liquidations'] + \
                                df_type['hourly_market-data_liquidations_all_exchange_all_symbol_short_liquidations']
            log_total_liquidation = np.log1p(total_liquidation)
            
            df_type.loc[:, 'scaled_liquidation_diff'] = scaler.fit_transform(liquidation_diff.values.reshape(-1, 1))
            df_type.loc[:, 'scaled_log_total_liquidation'] = scaler.fit_transform(log_total_liquidation.values.reshape(-1, 1))

        # train_df와 test_df를 다시 합치기
        self.df = pd.concat([train_df, test_df])

    
    def get_train_test(self):
        return (
        self.df.loc[self.df["_type"] == "train"],
        self.df.loc[self.df["_type"] == "test"],
        )

    def get_data(self) -> pd.DataFrame:
        return self.df



