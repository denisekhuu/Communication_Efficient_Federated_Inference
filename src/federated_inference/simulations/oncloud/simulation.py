import os
import json 

from federated_inference.common.environment import  DataMode, TransformType
from federated_inference.simulations.simulation import Simulation
from federated_inference.simulations.utils import *
from federated_inference.configs.model_configs import OnCloudModelConfiguration
from federated_inference.simulations.oncloud.client import OnCloudVerticalClient
from federated_inference.simulations.oncloud.server import OnCloudVerticalServer

class OnCloudVerticalSimulation(Simulation): 
    def __init__(self, seed, version, data_config, transform_config, model, transform_type: TransformType = TransformType.FULL_STRIDE_PARTITION, exist=False):
        self.seed = seed
        self.version = version
        self.data_config = data_config
        self.transform_config = transform_config
        self.server_model_config = OnCloudModelConfiguration
        self.data_mode = DataMode.VERTICAL
        self.transform_type = transform_type
        self.dataset =  self.load_data(data_config)
        self.client_datasets, self.transformation = self.transform_data(self.dataset, data_mode = self.data_mode, transform_config = transform_config, transform_type = self.transform_type)
        self.clients = [OnCloudVerticalClient(idx, dataset, data_config.LABELS) for idx, dataset in enumerate(self.client_datasets)]
        self.server = OnCloudVerticalServer(0, seed, OnCloudModelConfiguration(seed, version, model), self.data_config)
        self.cost_summary = {
            'overall_cost' : 0, 
            'reasons': []
        }

        if exist:
            self.server.load()
            self.load()

    def train(self): 
        datasets = [client.send_all() for client in self.clients]
        testsets = [client.request_pred(pred_all = True, keep_label = True) for client in self.clients]
        self.server.run_training(datasets, testsets)

    def load(self):
        from io import StringIO

        result_path = f"./results/oncloud/{self.data_config.DATASET_NAME}/{self.version}/{self.seed}/simulation.json"

        if not os.path.isfile(result_path):
            raise FileNotFoundError(f"No results file found at: {result_path}")

        with open(result_path, 'r') as f:
            results = json.load(f)

        self.results = results

        # Load server data
        if 'server' in results:
            self.server.train_losses = results['server'].get('training_losses', [])
            self.server.test_losses = results['server'].get('test_losses', [])
        else:
            print("[Warning] No 'server' key found in results.")



    def test_inference(self):
        testsets = [client.request_pred(pred_all = True, keep_label = True) for client in self.clients]
        predictions = self.server.pred(testsets)
        self.clients[0].check(predictions)
        
    def run_inference(self, testsets):
        predictions = self.server.pred(testsets)
        return predictions