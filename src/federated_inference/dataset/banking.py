import pandas as pd
from ucimlrepo import fetch_ucirepo
from .dataset import Dataset
from collections.abc import Iterable
from sklearn.model_selection import train_test_split

class BankMarketingDataset(Dataset):

    def __init__(self, config, transform_methods: Iterable = []):
        super(BankMarketingDataset, self).__init__(config, transform_methods)
        self.name = "Bank Marketing"
        self.dataset = fetch_ucirepo(id=222)
        self.X = self.dataset.data.features
        self.y = self.dataset.data.targets
        self.labels = self.y.unique().tolist()

    def load_train_data(self, test_size=0.2, random_state=42):
        X_train, _, y_train, _ = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        if self.transform_methods:
            for transform in self.transform_methods:
                X_train = transform(X_train)

        print("Bank Marketing training data loaded.")
        return X_train, y_train

    def load_test_data(self, test_size=0.2, random_state=42):
        _, X_test, _, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        if self.transform_methods:
            for transform in self.transform_methods:
                X_test = transform(X_test)

        print("Bank Marketing test data loaded.")
        return X_test, y_test
