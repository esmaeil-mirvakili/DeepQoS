import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from data.dataset import IOBinClassificationDataSet


class IONETLogisticRegression:
    def __init__(self, path, seed=42):
        self.path = path
        self.seed = seed
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.reset_model()

    def reset_model(self):
        self.model = LogisticRegression(random_state=self.seed, penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, tol=1e-4)

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
