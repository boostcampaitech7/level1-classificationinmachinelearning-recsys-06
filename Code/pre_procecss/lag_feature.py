import pandas as pd
from Code.pre_procecss.interface import PreProcessInterface


class PreProcessor(PreProcessInterface):
    
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self._preprocess()

    # 시차 (Lag) 특성 생성 메서드
    def create_lag_features(self, columns, max_lag):
        for col in columns:
            for lag in range(1, max_lag + 1):
                self.df[f'{col}_lag_{lag}'] = self.df[col].shift(lag)
        return self.df

    # 추가적인 시계열 특성 생성 메서드
    def create_moving_average_features(self, columns, windows):
        for col in columns:
            for window in windows:
                self.df[f'{col}_ma_{window}'] = self.df[col].rolling(window=window).mean() # 이동 평균
                self.df[f'{col}_std_{window}'] = self.df[col].rolling(window=window).std() # 이동 평균 편차
                self.df[f'{col}_diff_{window}'] = self.df[col] - self.df[col].shift(window) # 차분
                self.df[f'{col}_ema_{window}'] = self.df[col].ewm(span=window, adjust=False).mean() #지수 이동 평균
        return self.df

    # 결측값 제거 메서드
    def remove_ts_related_missing_values(self, columns, max_lag, windows):
        lagged_columns = [f'{col}_lag_{lag}' for col in columns for lag in range(1, max_lag + 1)]
        moving_avg_columns = [f'{col}_ma_{window}' for col in columns for window in windows]
        volatility_columns = [f'{col}_std_{window}' for col in columns for window in windows]
        difference_columns = [f'{col}_diff_{window}' for col in columns for window in windows]
        ema_columns = [f'{col}_ema_{window}' for col in columns for window in windows]
        
        relevant_columns = lagged_columns + moving_avg_columns + volatility_columns + difference_columns + ema_columns
        self.df = self.df.dropna(subset=relevant_columns).reset_index(drop=True)
        return self.df

    # 기본 전처리 실행 메서드 (사용자 정의 가능)
    def _preprocess(self):
        pass
