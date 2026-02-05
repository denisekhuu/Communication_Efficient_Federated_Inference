from torch.utils.data import Dataset as TorchDataset
from federated_inference.common.cost_calculator import CostCalculator
from federated_inference.common.environment import Member
from collections.abc import Iterable
import logging 

class NaiveVerticalClient():
    def __init__(self, 
        idx,
        dataset: TorchDataset,
        labels
    ):
        self.idx = idx
        self.data = dataset
        self.labels = labels
        self.numerical_labels = range(len(labels))
        self.member_type = Member.CLIENT
        self.costs = []

    def select_subset(self, ids: Iterable[int], set_type: str = "train"):
        if set_type == "test":
            return Subset(self.data.test_dataset, ids)
        else: 
            return Subset(self.data.train_dataset, ids)

    def send_all(self):
        cost = CostCalculator.calculate_communication_cost_by(self.data.train_dataset)
        cost.set_cost_reason("send_all_training")
        self.costs.append(cost)
        return self.data.train_dataset

    def request_pred(self, idx: int|None = None, set_type: str = "test", pred_all: bool= False, keep_label: bool = False): 
        if idx != None:
            if set_type == "test":
                data = self.data.test_dataset[idx]
                cost = CostCalculator.calculate_communication_cost_by(data)
                cost.set_cost_reason("send_testing_pred_request")
                self.costs.append(cost)
                
                return self.data.test_dataset[idx] if keep_label else self.data.test_dataset[idx][0] 
        elif pred_all:
            cost = CostCalculator.calculate_communication_cost_by(self.data.test_dataset)
            cost.set_cost_reason("send_all_testing_pred_request")
            self.costs.append(cost)
            return self.data.test_dataset if keep_label else [img for img, label in self.data.test_dataset]

    def check(self, predicted_labels,  pred_all: bool= True):
        from sklearn.metrics import accuracy_score, precision_score, recall_score,  confusion_matrix
        import pandas as pd
        if pred_all:
            true_labels = self.data.test_dataset.targets
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='macro')  # or 'weighted'
            recall = recall_score(true_labels, predicted_labels, average='macro')

            print("\n=== Metrics ===")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            cm = confusion_matrix(self.data.test_dataset.targets, predicted_labels, labels=self.numerical_labels)
            self.cm = pd.DataFrame(cm, index=[f'True {l}' for l in self.labels],
                                    columns=[f'Pred {l}' for l in self.labels])
            self.accuracy = accuracy 
            self.precision = precision 
            self.recall = recall