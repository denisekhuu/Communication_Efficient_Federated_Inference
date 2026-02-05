            
import torch
import numpy as np
from collections.abc import Iterable
import copy
from federated_inference.simulations.isolated.members.client import IsolatedVerticalClient
from federated_inference.simulations.simulation import Simulation
import math
from federated_inference.common.environment import TransformType, DataMode  
from federated_inference.simulations.isolated.configs.model_config import ModelConfiguration
from federated_inference.simulations.isolated.configs.data_config import DataConfiguration
from federated_inference.simulations.isolated.configs.transform_config import DataTransformConfiguration
import torch.nn as nn
import os
import json
import pandas as pd
from ..utils import *

class IsolatedVerticalSimulation(Simulation): 
    def __init__(self, 
                 data_config: DataConfiguration, 
                 transform_config: DataTransformConfiguration, 
                 seed: int, 
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
        self.clients = [IsolatedVerticalClient(idx, 
                                               seed, 
                                               ModelConfiguration(model, n_epochs=55), 
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
            
    def load(self):
        from io import StringIO
        result_path = f"./results/isolated/{self.data_config.DATASET_NAME}/{self.seed}/simulation.json"
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

    def collect_results(self, name: str, save: bool = True, figures: bool = False):
        import json
        import os
        from IPython.display import display

        self.results = {
            'seed': name,
            'clients': []
        }

        if figures:
            fig = create_simulation_image_subplots(self)
            display(fig)

        for client in self.clients:
            client_result = self._gather_client_results(client, figures)
            self.results['clients'].append(client_result)

        if save:
            self._save_results(name)
        return self.results

    def _gather_client_results(self, client, figures):
        result = {
            'idx': client.idx,
            'cm': client.cm.to_json(orient='split'),
            'training_losses': client.train_losses,
            'test_losses': client.test_losses
        }

        analysis, df_cm_per = cm_analysis(client)
        result['cm_analysis'] = analysis

        indices = [
            analysis['correct']['most_correct_class'],
            analysis['wrong']['most_misclassified_class']
        ]
        indices += [i for i in [analysis['wrong']['wrong_from'], analysis['wrong']['wrong_to']] if i not in indices]

        # Performance metrics
        result.update({
            'accuracy': analysis["performance"]["accuracy"],
            'precision': analysis["performance"]["precision"],
            'recall': analysis["performance"]["recall"]
        })

        # Generate and display figures if needed
        if figures:
            display(plot_test_loss(client.test_losses, 1, client.idx, "Test"))
            fig, selected_indices = create_client_image_subplots(self, [client.idx], 8, keys=indices)
            display(fig)
            result['client_image_subplots_ids'] = selected_indices
            display(print_cm_heat(df_cm_per, client.idx))

        return result

    def _save_results(self, name):
        import os
        import json

        result_path = f"./results/isolated/{self.data_config.DATASET_NAME}/{name}"
        os.makedirs(result_path, exist_ok=True)
        with open(os.path.join(result_path, "simulation.json"), "w") as f:
            json.dump(self.results, f, indent=4)
        print("Results saved to JSON.")


if __name__ == "__main__": 
    from federated_inference.simulations.isolated.configs.data_config import DataConfiguration
    from federated_inference.simulations.isolated.configs.transform_config import DataTransformConfiguration
    from federated_inference.simulations.isolated.configs.model_config import ModelConfiguration
    from federated_inference.simulations.isolated.models.IsolatedMnistModel import IsolatedMNISTModel
    from federated_inference.simulations.isolated.simulation import IsolatedVerticalSimulation


    data_config = DataConfiguration()
    transform_config = DataTransformConfiguration()
    model_config = ModelConfiguration(IsolatedMNISTModel)
    simulation = IsolatedVerticalSimulation(data_config, transform_config)
    simulation.train()
    simulation.test()
    print(simulation.to_json())
