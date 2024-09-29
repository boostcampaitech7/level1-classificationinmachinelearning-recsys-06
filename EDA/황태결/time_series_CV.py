import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


class TimeSeriesCV:
    def __init__(self, train_df):
        # 특징 및 타겟 정의
        self.model = None
        self.f1 = None
        self.cm = None
        self.recall = None
        self.precision = None
        self.y_pred_fold = None
        self.accuracy = None
        self.fold_results = None
        self.y_test_fold = None
        self.y_train_fold = None
        self.X_train_fold = None
        self.X_test_fold = None
        self.X = train_df.drop(["target", "ID"], axis=1)
        self.y = train_df["target"].astype(int)

        # LightGBM Parameters
        self.params = {
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": 4,
            "num_leaves": 50,
            "learning_rate": 0.05,
            "n_estimators": 30,
            "random_state": 42,
            "verbose": -1,
        }
        '''
            "boosting_type": "gbdt",
            "objective": "multiclass",
            "metric": "multi_logloss",
            "num_class": 4,
            "num_leaves": 31,
            "learning_rate": 0.01,
            "n_estimators": 100,
            "min_data_in_leaf": 20,
            "random_state": 42,
            "verbose": -1,
            'class_weight': 'balanced'
        }'''

        # TimeSeriesSplit configuration
        self.tscv = TimeSeriesSplit(n_splits=5)

        # Initialize results
        self.results = []

    def train(self):
        # TimeSeriesSplit loop
        for train_index, test_index in self.tscv.split(self.X):
            # Split the data
            self.X_train_fold, self.X_test_fold = self.X.iloc[train_index], self.X.iloc[test_index]
            self.y_train_fold, self.y_test_fold = self.y.iloc[train_index], self.y.iloc[test_index]

            # Train the LightGBM model
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(self.X_train_fold, self.y_train_fold)

            # Predict on test fold
            self.y_pred_fold = self.model.predict(self.X_test_fold)

            # Evaluate and store accuracy
            self.accuracy = accuracy_score(self.y_test_fold, self.y_pred_fold)
            self.results.append(self.accuracy)
            print(f"Fold accuracy: {self.accuracy}")

        # Final results for each fold
        self.fold_results = pd.DataFrame({"Fold": [f"Fold {i + 1}" for i in range(len(self.results))], "Accuracy": self.results})
        print(self.fold_results)
        self.cal_f1_score()
        self.cal_pr()
        self.cal_cm()

    def cal_f1_score(self):
        self.f1 = f1_score(self.y_test_fold, self.y_pred_fold, average='macro')
        return self.fold_results

    # Precision & Recall
    def cal_pr(self):
        self.precision = precision_score(self.y_test_fold, self.y_pred_fold, average='macro')
        self.recall = recall_score(self.y_test_fold, self.y_pred_fold, average='macro')
        return self.precision, self.recall

    # Confusion Matrix
    def cal_cm(self):
        self.cm = confusion_matrix(self.y_test_fold, self.y_pred_fold)
        return self.cm

    def print_performance(self):
        #print(f'fold_results:\n{self.fold_results}')
        print(f'F1 Score: {self.f1}')
        print(f'Precision: {self.precision}')
        print(f'Recall: {self.recall}')
        print(f'Confusion Matrix: \n{self.cm}')