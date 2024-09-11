from abc import abstractmethod, ABC

import pandas as pd


class PreProcessInterface(ABC):
    @abstractmethod
    def __init__(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def get_train_test(self):
        pass

    @abstractmethod
    def _preprocess(self):
        pass
