import torch
from collections.abc import Iterable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from torch.utils.data import Subset, DataLoader
from torch.utils.data import Dataset as TorchDataset
import os 


from federated_inference.common.environment import Member
from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.model_configs import  HybridSplitModelConfiguration


class HybridSplitClient():

    def __init__(self, 
            idx, 
            seed: int,
            data_config: DataConfiguration,
            model_config: HybridSplitModelConfiguration,
            dataset: TorchDataset,
            labels,
            log: bool = True, 
            log_interval: int = 100, 
            save_interval: int = 10
        ):
        self.idx = idx
        self.seed = seed
        self.data = dataset
        self.data_config = data_config
        self.model_config = model_config
        self.device = model_config.DEVICE
        self.labels = labels
        self.numerical_labels = range(len(labels))
        self.member_type = Member.CLIENT
        self.model = None
        self.log = log
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.base_model = model_config.CLIENT_BASE_MODEL().to(self.device)
        self.classifier_model = model_config.CLIENT_CLASSIFIER_MODEL().to(self.device)
        self.router_model = model_config.CLIENT_IG_MODEL().to(self.device)

    def select_subset(self, ids: Iterable[int], set_type: str = "train"):
        if set_type == "test":
            return Subset(self.data.test_dataset, ids)
        else: 
            return Subset(self.data.train_dataset, ids)

    def send_all(self):
        return self.data.train_dataset

    def request_pred(self, idx: int|None = None, set_type: str = "test", pred_all: bool= False, keep_label: bool = False): 
        if idx != None:
            if set_type == "test":
                return self.data.test_dataset[idx] if keep_label else self.data.test_dataset[idx][0] 
        elif pred_all:
            return self.data.test_dataset if keep_label else [img for img, label in self.data.test_dataset]

    def check(self, predicted_labels, pred_all: bool = True):
        if pred_all:
            true_labels = self.data.test_dataset.targets
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='macro')
            recall = recall_score(true_labels, predicted_labels, average='macro')
            f1 = f1_score(true_labels, predicted_labels, average='macro') 

            print("\n=== Metrics ===")
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}") 

            cm = confusion_matrix(true_labels, predicted_labels, labels=self.numerical_labels)
            self.cm = pd.DataFrame(cm, index=[f'True {l}' for l in self.labels],
                                    columns=[f'Pred {l}' for l in self.labels])
            self.accuracy = accuracy
            self.precision = precision
            self.recall = recall
            self.f1 = f1 


    def load(self):
        result_path = f"./results/hybrid/{self.model_config.version}/{self.data_config.DATASET_NAME}/{self.seed}"
        client_result_path = os.path.join(result_path, "clients") 

        # Base
        model_path = os.path.join(client_result_path, f'model_client_base_{self.idx}.pth').replace("\\", "/")
        network_state_dict = torch.load(model_path)
        self.base = self.base_model.load_state_dict(network_state_dict)

        # Classifier Head
        model_path = os.path.join(client_result_path, f'model_client_classifier_{self.idx}.pth').replace("\\", "/")
        network_state_dict = torch.load(model_path)
        self.classifier_model.load_state_dict(network_state_dict)

        # Router Head
        model_path = os.path.join(client_result_path, f'model_client_router_{self.idx}.pth').replace("\\", "/")
        network_state_dict = torch.load(model_path)
        self.router_model.load_state_dict(network_state_dict)

    def to_loader(self, testdata):
        self.testloader = DataLoader(self.data.test_dataset, batch_size=self.model_config.BATCH_SIZE_TEST, shuffle=False) 

            
