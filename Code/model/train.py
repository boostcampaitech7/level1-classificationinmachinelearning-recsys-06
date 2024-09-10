from typing import List
import pandas as pd

class Train():
    def __init__(self,x_train:pd.DataFrame, y_train:pd.DataFrame ,model_type:str,
                  learning_rate:float, random_state:int =42):
        # 하이퍼 파라미터 부분
        # 추가적인 하이퍼 파라미터 설정 가능 및 **kwarg로 추가 가능 해당 부분 예시
        # 추후 협의 필요.
        self.x_train = x_train
        self.y_train = y_train
        self.model_type = model_type
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # 학습을 위한 부분
        self._pre_process()
        self._train()
    
    def _train():
        '''
        학습 시키는 부분
        # example-1) 
        if model_type == "LSTM":
            LSTM.fit()
        '''
        return
    
    def _pre_process():
        '''
        from pre_procecss import PreProcess
        데이터 전처리 호출 부분
        self.x_train = PreProcess(self.x_train)
        '''