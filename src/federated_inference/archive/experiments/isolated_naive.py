from federated_inference.simulations.isolated.configs.data_config import DataConfiguration
from federated_inference.simulations.isolated.configs.transform_config import DataTransformConfiguration
from federated_inference.simulations.isolated.configs.model_config import ModelConfiguration
from federated_inference.simulations.simulation import Simulation
from federated_inference.simulations.isolated.models.IsolatedMnistModel import IsolatedMNISTModel
from federated_inference.simulations.isolated.models.IsolatedFmnistModel import IsolatedFMNISTModel
from federated_inference.simulations.isolated.simulation import IsolatedVerticalSimulation
from federated_inference.simulations.naive.configs.data_config import DataConfiguration
from federated_inference.simulations.naive.configs.transform_config import DataTransformConfiguration
from federated_inference.simulations.naive.configs.model_config import ModelConfiguration
from federated_inference.simulations.naive.models.NaiveMnistModel import NaiveMNISTModel
from federated_inference.simulations.naive.models.NaiveFmnistModel import NaiveFMNISTModel
from federated_inference.simulations.naive.simulation import NaiveVerticalSimulation

import torch 
import random 
import numpy as np

    
def set_seed(seed=42):
    torch.manual_seed(seed)                # CPU
    torch.cuda.manual_seed(seed)           # Current GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs
    np.random.seed(seed)                   # NumPy
    random.seed(seed)                      # Python random
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


class NaiveIsolatedExperiment():
    def _simulation_results(name='MNIST', models =[NaiveMNISTModel, IsolatedMNISTModel]):  
        seeds = [1,2,3,4,5]
        for seed in seeds:
            set_seed(seed)
            from federated_inference.simulations.isolated.configs.data_config import DataConfiguration
            data_config = DataConfiguration(name)
            transform_config = DataTransformConfiguration()
            simulation = NaiveVerticalSimulation(seed, data_config, transform_config, models[0], exist = False)
            simulation.train()
            simulation.test_inference()
            result = simulation.collect_results(seed, save = False)
            self.naive_results.append(result)
            from federated_inference.simulations.naive.configs.data_config import DataConfiguration
            data_config = DataConfiguration(name)
            simulation = IsolatedVerticalSimulation(data_config, transform_config, seed, models[1], exist=False)
            simulation.train()
            simulation.test()
            result = simulation.collect_results(seed, save = False, figures=True)
            self.isolated_results.append(result)

    def _precision_recall(self, idx):
        import numpy as np
        values = [r['client'][0]['precision'][idx] for r in self.naive_results]
        mean =  np.mean(values)
        var = np.var(values)
        var1 = (mean, var)
        var2 = []
        for i in range(len(isolated_results[0]['clients'])):
            values = [r['clients'][i]['precision'][idx] for r in self.isolated_results]
            mean =  np.mean(values)
            var = np.var(values)
            var2.append((mean, var))
        var4 = []
        values = [r['client'][0]['recall'][idx] for r in self.naive_results]
        mean =  np.mean(values)
        var = np.var(values)
        var3 = (mean, var)
        for i in range(len(isolated_results[0]['clients'])):
            values = [r['clients'][i]['recall'][idx] for r in self.isolated_results]
            mean =  np.mean(values)
            var = np.var(values)
            var4.append((mean, var))
        return var1, var2, var3, var4

    def __precision_recall_fig(self, name, idx):
        var1, var2, var3, var4 = self._precision_recall(idx)
        # Labels for x-axis
        labels = ["Naive Server"] + [f"Isolated Client {i}" for i in range(len(var2))]
        
        # Build traces
        fig = go.Figure()
        
        # Precision trace
        fig.add_trace(go.Bar(
            x=labels,
            y=[var1[0]] + [v[0] for v in var2],
            name='Precision',
            marker_color='steelblue',
            error_y=dict(
                type='data',
                array=[var1[1]**0.5] + [v[1]**0.5 for v in var2],
                visible=True
            )
        ))
        
        # Recall trace
        fig.add_trace(go.Bar(
            x=labels,
            y=[var3[0]] + [v[0] for v in var4],
            name='Recall',
            marker_color='darkorange',
            error_y=dict(
                type='data',
                array=[var3[1]**0.5] + [v[1]**0.5 for v in var4],
                visible=True
            )
        ))
        
        # Layout
        fig.update_layout(
            title=f"{name} Precision and Recall of Class {idx}",
            xaxis_title='Experiment',
            yaxis_title='Score',
            yaxis=dict(range=[0.2, 1.0]),
            barmode='group',
            template='plotly_white'
        
        )
        
        fig.show()

    def show_recall_figs(self):
        for i in range(10):
            self.__precision_recall_fig('Fashion MNIST',i)

    
    def _accuracy(self):
        import numpy as np
        
        values = [r['client'][0]['accuracy'] for r in self.naive_results]
        mean =  np.mean(values)
        var = np.var(values)
        var1 = (mean, var)
        
        var2 = []
        for i in range(len(isolated_results[0]['clients'])):
            values = [r['clients'][i]['accuracy'] for r in self.isolated_results]
            mean =  np.mean(values)
            var = np.var(values)
            var2.append((mean, var))
        return var1, var2
    
    def _accuracy_comparison(): 
        var1, var2 = self._accuracy()
        
        import plotly.graph_objects as go
        results = [("Naive Server", var1[0], var1[1])] + [(f"Isolated Client {i}", acc, var) for i, (acc, var) in enumerate(var2)]
        
        labels = [r[0] for r in results]
        accuracies = [r[1] for r in results]
        variances = [r[2] for r in results]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=labels,
            y=accuracies,
            name='Accuracy',
            marker_color='steelblue',
            error_y=dict(
                type='data',
                array=[v ** 0.5 for v in variances],  # Use stddev for error bars
                visible=True
            )
        ))
        s
        # Customize layout
        fig.update_layout(
            title='Fashion MNIST Accuracy',
            xaxis_title='Experiment',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0.7, 1.0]),
            template='plotly_white'
        )
        
        fig.show()
    _accuracy_comparison()
