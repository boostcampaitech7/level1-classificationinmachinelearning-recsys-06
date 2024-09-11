from typing import Type

import pandas as pd
from numpy import ndarray

from Code.model.baseline import LightGBM
from Code.model.interface import ModelInterface
from Code.pre_procecss.baseline import BaseLineData
from Code.pre_procecss.interface import PreProcessInterface


class Model:
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data
        self._get_config()  # 하이퍼 파라미터 부분 및 기타 설정 - config.yaml 수정에 따라 사용
        self.model_type = self.config.get("server").get("model_type")
        self.model: ModelInterface
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None
        # 데이터 전처리
        self._get_pre_process()
        # 학습을 위한 부분
        self._set_model_preprocess()

    def _get_config(self):
        from Code.config import get_config

        self.config = get_config()

    def _set_model_preprocess(self):
        if self.train_df is None or self.test_df is None:
            pre_precess = self._get_pre_process()(self.data)
            self.train_df, self.test_df = pre_precess.get_train_test()
        model = self._get_model()(
            self.train_df.drop(["target", "ID"], axis=1, errors="ignore"),
            self.train_df["target"],
            self.config,
        )
        self.model = model
        return

    def _get_model(self) -> Type[ModelInterface]:
        if self.model_type.lower() == "baseline":
            return LightGBM
        else:
            raise Exception(f"{self.model_type}: 해당 모델은 지원되지 않습니다.")

    def _get_pre_process(self) -> [PreProcessInterface]:
        if self.model_type.lower() == "baseline":
            return BaseLineData
        else:
            raise Exception(f"{self.model_type}: 해당 모델은 지원되지 않습니다.")

    def train(self):
        self.model.train()

    def train_for_validation(self):
        self.model.train_validation()

    def predict(self) -> ndarray | list:
        return self.model.predict(
            self.test_df.drop(["target", "ID"], axis=1, errors="ignore")
        )
