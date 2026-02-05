            
from federated_inference.common.environment import  DataMode, TransformType
from federated_inference.configs.model_configs import HybridSplitModelConfiguration
from federated_inference.simulations.simulation import Simulation
from federated_inference.simulations.utils import *
from federated_inference.simulations.hybrid.server import HybridSplitServer
from federated_inference.simulations.hybrid.client import HybridSplitClient

class HybridSplitSimulation(Simulation): 
    def __init__(self, seed, version, data_config, transform_config, server_model, client_base_model, client_classifier_model, client_ig_model, transform_type: TransformType = TransformType.FULL_STRIDE_PARTITION, exist=False):
        self.seed = seed
        self.data_config = data_config
        self.transform_config = transform_config
        self.server_model_config = HybridSplitModelConfiguration(version, server_model, client_base_model, client_classifier_model, client_ig_model)
        self.data_mode = DataMode.VERTICAL
        self.transform_type = transform_type
        self.dataset =  self.load_data(data_config)
        self.client_datasets, self.transformation = self.transform_data(self.dataset, data_mode = self.data_mode, transform_config = transform_config, transform_type = self.transform_type)
        self.clients = [HybridSplitClient(idx, seed, data_config, self.server_model_config, dataset, data_config.LABELS) for idx, dataset in enumerate(self.client_datasets)]
        self.server = HybridSplitServer(0, seed, self.server_model_config , self.data_config)

    def train(self): 
        datasets = [client.send_all() for client in self.clients]
        testsets = [client.request_pred(pred_all = True, keep_label = True) for client in self.clients]
        self.server.run_training(datasets, testsets)

    def test_inference(self):
        testsets = [client.request_pred(pred_all = True, keep_label = True) for client in self.clients]
        predictions = self.server.pred(testsets)
        self.clients[0].check(predictions)