import torch.nn as nn
import os
import json
import pandas as pd
from federated_inference.simulations.utils import *
from federated_inference.simulations.simulation import Simulation
from federated_inference.common.environment import TransformType, DataMode  

from federated_inference.simulations.ondevice.client import OnDeviceVerticalClient
from federated_inference.configs.model_configs import OnDeviceModelConfiguration

class OnDeviceVerticalSimulation(Simulation): 
    def __init__(self, 
                 seed: int,
                 version: str,
                 data_config, 
                 transform_config, 
                 model: nn.Module,
                 transform_type:  TransformType = TransformType.FULL_STRIDE_PARTITION, 
                 exist = False):

        self.data_config = data_config
        self.transform_config = transform_config
        self.seed = seed
        self.data_mode = DataMode.VERTICAL
        self.transform_type = transform_type
        self.dataset =  self.load_data(data_config)
        self.client_datasets, self.transformation = self.transform_data(self.dataset, data_mode = self.data_mode, transform_config = transform_config, transform_type = self.transform_type)
        self.clients = [OnDeviceVerticalClient(idx, 
                                               seed, 
                                               OnDeviceModelConfiguration(version, model), 
                                               data_config,
                                               dataset, 
                                               data_config.LABELS) for idx, dataset in enumerate(self.client_datasets)]

        if exist:
            for client in self.clients:
                client.load()
            self.load()
                
    def train(self): 
        for client in self.clients:
            client.run_training()

    def test(self):
        for client in self.clients:
            predictions = client.pred()
            client.check(predictions)

    def run_inference(self, _testsets):
        predictions = {}
        for client, testset in zip(self.clients, _testsets):
            predictions[client.idx] = client._pred(testset)
        return predictions

            
    def load(self):
        from io import StringIO
        result_path = f"./results/ondevice/{self.data_config.DATASET_NAME}/{self.seed}/simulation.json"
        if os.path.isfile(result_path):
            with open(result_path, 'r') as f:
                results = json.load(f)
    
            self.results = results 
            for client_data in results['clients']:
                idx = client_data['idx']
                client = next((c for c in self.clients if c.idx == idx), None)
                if client:
                    client.cm = pd.read_json(StringIO(client_data['cm']), orient='split')
                    client.train_losses = client_data['training_losses']
                    client.test_losses = client_data['test_losses']
                else:
                    print(f"[Warning] No client with idx={idx} found in self.clients.")
        
    def to_json(self):
        import json
        simulation_data = {
            "configs" : {
                "data": self.data_config.__dict__(),
                "data_mode" : self.data_mode.value,
                "transformation": {
                    "transformation_type": self.transform_type.value,
                    "transoformation_config": self.transform_config.__dict__()
                }
            }

        }
        return json.dumps(simulation_data)
