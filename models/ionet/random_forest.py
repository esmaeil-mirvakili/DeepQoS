import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data.dataset import IOBinClassificationDataSet


class IONETRandomForest:
    def __init__(self, path, n_estimators=100, max_depth=20, seed=42):
        self.path = path
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.seed = seed
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.reset_model()

    def reset_model(self):
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=self.seed, criterion='gini',
                                  min_samples_split=5, min_samples_leaf=2,
                                  max_features='sqrt', bootstrap=True,)

    def load_data(self):
        self.train_dataset = IOBinClassificationDataSet(self.path, stage='train')
        self.test_dataset = IOBinClassificationDataSet(self.path, stage='test')

    def train(self):
        X_train = np.array([self.train_dataset[i][0] for i in range(len(self.train_dataset))])  # Features
        y_train = np.array([self.train_dataset[i][1] for i in range(len(self.train_dataset))])  # Labels
        self.model.fit(X_train, y_train)

    def test(self):
        X_test = np.array([self.test_dataset[i][0] for i in range(len(self.test_dataset))])  # Features
        y_test = np.array([self.test_dataset[i][1] for i in range(len(self.test_dataset))])  # Labels
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report
