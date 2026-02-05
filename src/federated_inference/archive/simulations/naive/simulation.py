

from federated_inference.common.environment import  DataMode, DataSetEnum, TransformType
from federated_inference.simulations.simulation import Simulation
from federated_inference.simulations.naive.configs.model_config import ModelConfiguration
from .members.client import NaiveVerticalClient
from .members.server import NaiveVerticalServer
from ..utils import *
import os
import json 
import pandas as pd

class NaiveVerticalSimulation(Simulation): 
    def __init__(self, seed, data_config, transform_config, model, transform_type: TransformType = TransformType.FULL_STRIDE_PARTITION, exist=False):
        self.seed = seed
        self.data_config = data_config
        self.transform_config = transform_config
        self.server_model_config = ModelConfiguration
        self.data_mode = DataMode.VERTICAL
        self.transform_type = transform_type
        self.dataset =  self.load_data(data_config)
        self.client_datasets, self.transformation = self.transform_data(self.dataset, data_mode = self.data_mode, transform_config = transform_config, transform_type = self.transform_type)
        self.clients = [NaiveVerticalClient(idx, dataset, data_config.LABELS) for idx, dataset in enumerate(self.client_datasets)]
        self.server = NaiveVerticalServer(0, seed, ModelConfiguration(model), self.data_config)
        self.cost_summary = {
            'overall_cost' : 0, 
            'reasons': []
        }

        if exist:
            self.server.load()
            self.load()

    def train(self): 
        self.request_training()
        datasets = [client.send_all() for client in self.clients]
        testsets = [client.request_pred(pred_all = True, keep_label = True) for client in self.clients]
        self.server.run_training(datasets, testsets)

    def load(self):
        from io import StringIO

        result_path = f"./results/naive/{self.data_config.DATASET_NAME}/{self.seed}/simulation.json"

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

        # Load client data
        for client_data in results.get('client', []):
            idx = client_data['idx']
            client = next((c for c in self.clients if c.idx == idx), None)

            if client:
                # Load confusion matrix
                client.cm = pd.read_json(StringIO(client_data['cm']), orient='split')
                # Load losses
                client.train_losses = client_data.get('training_losses', [])
                client.test_losses = client_data.get('test_losses', [])
            else:
                print(f"[Warning] No client with idx={idx} found in self.clients.")

    def test_inference(self):
        testsets = [client.request_pred(pred_all = True, keep_label = True) for client in self.clients]
        predictions = self.server.pred(testsets)
        self.clients[0].check(predictions)
        
    def show_cost_summary(self):
        import json
        self.cost_summary['overall_cost'] = self.cost_summary['overall_cost'] +  len(self.server.costs)
        self.cost_summary['reasons']+=[cost.__dict__() for cost in self.server.costs]
        for client in self.clients:
            self.cost_summary['overall_cost'] = self.cost_summary['overall_cost'] +  len(client.costs)
            self.cost_summary['reasons']+=([cost.__dict__() for cost in client.costs])
        return json.dumps(self.cost_summary)

    def request_training(self):
        # Active client or server request training
        self.cost_summary['overall_cost'] = self.cost_summary['overall_cost'] + len(self.clients)
        self.cost_summary['reasons']+=[{"reason": 'requested, training'}]

    def to_json(self):
        import json
        simulation_data = {
            "configs" : {
                "data": self.data_config.__dict__(),
                "data_mode" : self.data_mode.value,
                "transformation": {
                    "transformation_type": self.transform_type.value,
                    "transoformation_config": self.transform_config.__dict__()
                },
                "server_model": self.server_model_config.__dict__()
            }

        }
        return json.dumps(simulation_data)

    def collect_results(self, name: str, save: bool = True, figures: bool = False):
        from IPython.display import display
        import json
        import os

        self.results = {
            'seed': name,
            'client': [],
            'server': {}
        }

        if figures:
            fig = create_simulation_image_subplots(simulation)
            display(fig)

        # Collect server results
        self.results['server'] = self._gather_server_results(self.server)

        # Collect results for the first (and only) client
        client = self.clients[0]
        client_result = self._gather_client_result(client, figures)
        self.results['client'].append(client_result)

        # Save results
        if save:
            self._save_results(str(name), base_dir="naive")
        return self.results

    def _gather_server_results(self, server):
        return {
            'training_losses': server.train_losses,
            'test_losses': server.test_losses
        }

    def _gather_client_result(self, client, figures):
        from IPython.display import display

        result = {
            'idx': client.idx,
            'cm': client.cm.to_json(orient='split')
        }

        analysis, df_cm_per = cm_analysis(client)
        result['cm_analysis'] = analysis

        # Extract relevant class indices
        indices = [
            analysis['correct']['most_correct_class'],
            analysis['wrong']['most_misclassified_class']
        ]
        indices += [i for i in [analysis['wrong']['wrong_from'], analysis['wrong']['wrong_to']] if i not in indices]

        # Add performance metrics
        result.update({
            'accuracy': analysis["performance"]["accuracy"],
            'precision': analysis["performance"]["precision"],
            'recall': analysis["performance"]["recall"]
        })

        # Generate figures if needed
        if figures:
            display(plot_test_loss(client.test_losses, 1, client.idx, "Test"))
            fig, subplot_indices = create_client_image_subplots(self, [client.idx], 8, keys=indices)
            display(fig)
            result['client_image_subplots_ids'] = subplot_indices
            fig = print_cm_heat(df_cm_per, client.idx)
            display(fig)

        return result

    def _save_results(self, name, base_dir="naive"):
        import os
        import json

        result_path = os.path.join("./results", base_dir , self.data_config.DATASET_NAME, name)
        os.makedirs(result_path, exist_ok=True)
        file_path = os.path.join(result_path, "simulation.json")

        with open(file_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print("Results saved to JSON.")


if __name__ == "__main__": 
    from federated_inference.simulations.naive.configs.data_config import DataConfiguration
    from federated_inference.simulations.naive.configs.transform_config import DataTransformConfiguration
    from federated_inference.simulations.naive.configs.model_config import ModelConfiguration
    from federated_inference.simulations.naive.models.NaiveMnistModel import NaiveMNISTModel
    from federated_inference.simulations.naive.simulation import NaiveVerticalSimulation
    data_config = DataConfiguration()
    transform_config = DataTransformConfiguration()
    server_model_config = ModelConfiguration(NaiveMNISTModel)
    simulation = NaiveVerticalSimulation(data_config, transform_config, server_model_config)
    simulation.train()
    simulation.test_inference()
    print(simulation.show_cost_summary())
    print(simulation.to_json())

