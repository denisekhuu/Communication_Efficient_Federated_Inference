from federated_inference.configs.data_config import DataConfiguration
from federated_inference.configs.torch_config import TorchConfiguration
from federated_inference.configs.env_config import EnvConfiguration
from federated_inference.configs.model_config import ModelConfiguration
from federated_inference.configs.member_config import MemberConfiguration
from torch.utils.data import Subset, Dataset
from collections.abc import Iterable
from federated_inference.common.cost_calculator import CostCalculator
from federated_inference.common.env_utils import EnvUtils

class CLClient():
    def __init__(self, 
        dataset: Dataset,
        data_config: DataConfiguration, 
        torch_config: TorchConfiguration, 
        env_config: EnvConfiguration, 
        model_config: ModelConfiguration, 
        member_config: MemberConfiguration,
        env_utils: EnvUtils = None,
    ):
        self.data = dataset

    def select_subset(self, ids: Iterable[int], set_type: str = "train"):
        if set_type == "test":
            return Subset(self.data.test_dataset, ids)
        else: 
            return Subset(self.data.train_dataset, ids)

    def send_all(self, calc_cost: bool = True):
        if calc_cost:
            CostCalculator.calculate_communication_cost_by(self.data.train_dataset)
        return self.data.train_dataset

    def request_pred(self, idx: int|None = None, set_type: str = "test", pred_all: bool= False, keep_label: bool = False): 
        if idx != None:
            if set_type == "test":
                data = self.data.test_dataset[idx]
                CostCalculator.calculate_communication_cost_by(data)
                return self.data.test_dataset[idx] if keep_label else self.data.test_dataset[idx][0] 
        elif pred_all:
            CostCalculator.calculate_communication_cost_by(self.data.test_dataset)
            return self.data.test_dataset if keep_label else [img for img, label in self.data.test_dataset]
