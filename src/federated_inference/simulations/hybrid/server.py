import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from federated_inference.common.environment import Member
from federated_inference.common.early_stopper import EarlyStopper

class HybridSplitServer():

    def __init__(self, 
            idx: int, 
            seed: int,
            model_config,
            data_config,
            log: bool = True, 
            log_interval: int = 100,
            save_interval: int = 20
        ):
        self.idx = idx
        self.seed = seed
        self.version = model_config.version
        self.model_config = model_config
        self.data_config = data_config
        self.seed = seed
        self.number_of_clients = 4
        self.n_epoch = model_config.N_EPOCH
        self.device = model_config.DEVICE
        self.member_type = Member.SERVER
        self.server_model = model_config.SERVER_MODEL().to(self.device)
        self.client_base_models = [model_config.CLIENT_BASE_MODEL().to(self.device) for c in range(self.number_of_clients)]
        self.client_classifier_models = [model_config.CLIENT_CLASSIFIER_MODEL().to(self.device) for c in range(self.number_of_clients)]
        self.client_ig_models = [model_config.CLIENT_IG_MODEL().to(self.device) for c in range(self.number_of_clients)]
        self.CRITERION = nn.CrossEntropyLoss()
        self.CRITERION_NAME = "CrossEntropyLoss"
        self.weight_decay = 0
        self.SERVER_OPTIMIZER = optim.Adam(self.server_model.parameters() , lr=model_config.LEARNING_RATE, weight_decay=self.weight_decay)
        self.CLIENT_BASE_OPTIMIZERS =  [optim.Adam(self.client_base_models[c].parameters() , lr=model_config.LEARNING_RATE, weight_decay=self.weight_decay ) for c in range(self.number_of_clients)]
        self.CLIENT_CLASSIFIER_OPTIMIZERS =  [optim.Adam(self.client_classifier_models[c].parameters() , lr=model_config.LEARNING_RATE, weight_decay=self.weight_decay) for c in range(self.number_of_clients)]
        self.CLIENT_IG_OPTIMIZERS =  [optim.Adam(self.client_ig_models[c].parameters() , lr=model_config.LEARNING_RATE, weight_decay=self.weight_decay ) for c in range(self.number_of_clients)]
        self.OPTIMIZER_NAME = "Adam"
        self.LOCAL_CLASSIFIER_CRITERION = nn.CrossEntropyLoss()
        self.LOCAL_IG_CRITERION = nn.BCELoss()

        self.log = log
        self.log_interval = log_interval
        self.save_interval = save_interval

        if self.log: 
            self.train_losses = []
            self.test_losses = []
            self.accuracies = []


    def _to_loader(self, trainsets, testsets, batch_size_train, batch_size_val, batch_size_test, train_shuffle, val_shuffle, test_shuffle, train_ratio):
        # TODO refactoing to use self
        if True:
            dataset_length = len(trainsets[0])
            assert all(len(trainset) == dataset_length for trainset in trainsets), "All trainsets must be the same length"

            indices = np.arange(dataset_length)
            self.train_set_indices = np.arange(dataset_length)

            np.random.shuffle(indices)

            train_end = round(train_ratio * dataset_length)
            
            
            self.train_indices = indices[:train_end]
            val_indices = indices[train_end:]

            traindatas = [Subset(trainset, self.train_indices) for trainset in trainsets]
            valdatas = [Subset(trainset, val_indices) for trainset in trainsets]
        else:
            traindatas = [Subset(trainset, range(round(train_ratio*len(trainset)))) for trainset in trainsets]
            valdatas = [Subset(trainset, range(round(train_ratio*len(trainset)), len(trainset))) for trainset in trainsets]
        self.trainloader = [DataLoader(traindata, batch_size=batch_size_train, shuffle=False)  for traindata in traindatas]
        self.valloader = [DataLoader(valdata, batch_size=batch_size_val, shuffle=False) for valdata in valdatas]
        self.testloader = [DataLoader(testdata, batch_size=batch_size_test, shuffle=False) for testdata in testsets]

    def shuffle_loader(self, trainsets, batch_size_train, batch_size_val, train_ratio):
        # TODO refactoing to use self
        if True:
            dataset_length = len(trainsets[0])
            assert all(len(trainset) == dataset_length for trainset in trainsets), "All trainsets must be the same length"

            np.random.shuffle(self.train_set_indices)

            train_end = round(train_ratio * dataset_length)
            train_indices = self.train_set_indices[:train_end]
            val_indices = self.train_set_indices[train_end:]

            traindatas = [Subset(trainset, train_indices) for trainset in trainsets]
            valdatas = [Subset(trainset, val_indices) for trainset in trainsets]
        else:
            traindatas = [Subset(trainset, range(round(train_ratio*len(trainset)))) for trainset in trainsets]
            valdatas = [Subset(trainset, range(round(train_ratio*len(trainset)), len(trainset))) for trainset in trainsets]
        self.trainloader = [DataLoader(traindata, batch_size=batch_size_train, shuffle=False)  for traindata in traindatas]
        self.valloader = [DataLoader(valdata, batch_size=batch_size_val, shuffle=False) for valdata in valdatas]


    def _pred_loader(self, testsets, batch_size_test, test_shuffle):
        return  [DataLoader(testdata, batch_size=batch_size_test, shuffle=False) for testdata in testsets]
        
    
    def train(self, epoch):
        for batch_idx, batches in enumerate(zip(*self.trainloader)):
            data_slices = [batch[0].to(self.device).float() for batch in batches]
            target = batches[0][1].to(self.device).long()
    
            # ====== Base forward passes ======
            client_activations = []
            classifier_grads_per_client = []
            classifier_losses = []
            classifier_preds = []
    
            for data, base_model, classifier_model, optimizer in zip(data_slices, self.client_base_models,  self.client_classifier_models, self.CLIENT_CLASSIFIER_OPTIMIZERS):
                base_model.train()
                activation = base_model(data)
                activation.requires_grad_()
                client_activations.append(activation)
                classifier_model.train()
                optimizer.zero_grad()
                classifier_output = classifier_model(activation)
                classifier_pred = classifier_output.argmax(dim=1)
                loss = self.LOCAL_CLASSIFIER_CRITERION(classifier_output, target)
                classifier_losses.append(loss.item())
    
                grads = torch.autograd.grad(loss, [activation] + list(classifier_model.parameters()), retain_graph=True)
                classifier_grads_per_client.append(grads[0])
    
                for param, grad in zip(classifier_model.parameters(), grads[1:]):
                    param.grad = grad
                optimizer.step()
    
                classifier_preds.append(classifier_pred)
            concat_activations = torch.cat(client_activations, dim=1)
    
            # ====== Server forward and backprop======
            self.server_model.train()
            self.SERVER_OPTIMIZER.zero_grad()
            server_output = self.server_model(concat_activations)
            server_loss = self.CRITERION(server_output, target)
            server_pred = server_output.argmax(dim=1)
            server_grads = torch.autograd.grad(server_loss, [concat_activations] + list(self.server_model.parameters()), retain_graph=True)
            #Server Backprop
            server_concat_activation_grad = server_grads[0]
            server_model_grads = server_grads[1:]
            
            # Apply gradients manually to model parameters
            for param, grad in zip(self.server_model.parameters(), server_model_grads):
                param.grad = grad  # Set .grad for optimizer
    
            self.SERVER_OPTIMIZER.step()
    

            # ====== IG forward and backprop ======
            ig_grads_per_client = []
            for i, (activation, ig_model, ig_optimizer) in enumerate(zip(client_activations, self.client_ig_models, self.CLIENT_IG_OPTIMIZERS)):
                ig_model.train()
                ig_optimizer.zero_grad()
            
                # Get IG prediction
                ig_output = ig_model(activation).squeeze(1)  # (B,)
            
                client_wrong = (classifier_pred != target)
                server_right = (server_pred == target)
                ig_target = (client_wrong & server_right).float() # Shape: (batch_size, 1)
                ig_loss = self.LOCAL_IG_CRITERION(ig_output, ig_target)

                grads = torch.autograd.grad(ig_loss, [activation] + list(ig_model.parameters()))
                ig_grads_per_client.append(grads[0])
            
                for param, grad in zip(ig_model.parameters(), grads[1:]):
                    param.grad = grad
                ig_optimizer.step()
    
            # ====== Base Backprop combined gradients ======
            activation_sizes = [act.shape[1] for act in client_activations]
            activation_grads_server = torch.split(server_concat_activation_grad, activation_sizes, dim=1)
    
            for i, (base_model, optimizer, data) in enumerate(zip(self.client_base_models, self.CLIENT_BASE_OPTIMIZERS, data_slices)):
                optimizer.zero_grad()
    
                # Combine gradients: server + classifier + IG
                combined_grad = activation_grads_server[i] + classifier_grads_per_client[i] + ig_grads_per_client[i]
    
                # Backward through base
                activation = base_model(data)
                activation.backward(combined_grad)
                optimizer.step()
            
            if batch_idx % self.log_interval == 0:
                print(f"Epoch {epoch} | Batch {batch_idx} | Server Loss: {server_loss.item():.4f} | "
                      f"Classifier Losses: {[round(x, 4) for x in classifier_losses]}")

    def validate(self):
        self.server_model.eval()
        
        val_loss = 0
        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*self.valloader)):
                # batches is tuple of batch from each loader
                data_slices = [batch[0].to(self.device).float() for batch in batches]
                target = batches[0][1].to(self.device).long()
                client_activations = []
                client_val_loss = 0
                for data, client_base_model, classifier_model in zip(data_slices, self.client_base_models, self.client_classifier_models):
                    #Client FeedForward
                    client_base_model.eval()
                    activation = client_base_model(data)
                    client_activations.append(activation)
                    classifier_output = classifier_model(activation)
                    client_loss = self.LOCAL_CLASSIFIER_CRITERION(classifier_output, target).item()
                    client_val_loss += client_loss
                client_val_loss /= self.number_of_clients
                concat_activations = torch.cat(client_activations, dim=1)
                #Server FeedForward
                self.server_model.eval()
                output = self.server_model(concat_activations)
                val_loss +=  self.CRITERION(output, target).item()

        val_loss /= len(self.valloader[0].dataset)

        if self.early_stopper.best_loss is None or val_loss < self.early_stopper.best_loss:
            print("Validation loss improved. Saving model...")
            self.save()
        self.early_stopper(val_loss)

    def test(self):
        self.server_model.eval()
        
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*self.testloader)):
                # batches is tuple of batch from each loader
                data_slices = [batch[0].to(self.device).float() for batch in batches]
                target = batches[0][1].to(self.device).long()
                client_activations = []
                for data, client_base_model in zip(data_slices, self.client_base_models):
                    #Client FeedForward
                    client_base_model.eval()
                    activation = client_base_model(data)
                    client_activations.append(activation)
                concat_activations = torch.cat(client_activations, dim=1)
                #Server FeedForward
                self.server_model.eval()
                output = self.server_model(concat_activations)
                test_loss +=  self.CRITERION(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.testloader[0].dataset)
        accuracy = 100. * correct / len(self.testloader[0].dataset)
        if self.log: 
            self.test_losses.append(test_loss)
            self.accuracies.append(accuracy)
        print(f'\nTest set: Average loss per Sample: {test_loss:.4f}, Accuracy: {correct}/{len(self.testloader[0].dataset)} '
            f'({accuracy:.0f}%)\n')
            
        
    def test_inferences(self):
        self.server_model.eval()
        clients_preds = {i: [] for i in range(self.number_of_clients)}
        clients_router = {i: [] for i in range(self.number_of_clients)}
        test_loss = 0
        server_preds = []

        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*self.testloaders)):
                # batches is tuple of batch from each loader
                data_slices = [batch[0].to(self.device).float() for batch in batches]
                target = batches[0][1].to(self.device).long()
                client_activations = []
                for i, (data, client_base_model, classifier_model, router_model) in enumerate(zip(data_slices, self.client_base_models, self.client_classifier_models, self.client_ig_models)):
                    #Client FeedForward
                    client_base_model.eval()
                    classifier_model.eval()
                    router_model.eval()
                    activation = client_base_model(data)
                    client_class_logits = classifier_model(activation)
                    clients_preds[i] += client_class_logits.argmax(dim=1, keepdim=True).squeeze().tolist()
                    client_router_logits = router_model(activation)
                    clients_router[i] += client_router_logits.view(-1).tolist()
                    client_activations.append(activation)
                    
                concat_activations = torch.cat(client_activations, dim=1)
                #Server FeedForward
                self.server_model.eval()
                output = self.server_model(concat_activations)
                test_loss +=  self.CRITERION(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                server_preds += pred.squeeze().tolist()

        return clients_preds, clients_router, server_preds
            
    def run_training(self, trainset, testset):
        self.server_early_stopper = EarlyStopper()
        self.early_stopper = EarlyStopper()
        self._to_loader(trainset, testset, 
            self.model_config.BATCH_SIZE_TRAIN,
            self.model_config.BATCH_SIZE_VAL, 
            self.model_config.BATCH_SIZE_TEST, 
            self.model_config.TRAIN_SHUFFLE,
            self.model_config.VAL_SHUFFLE, 
            self.model_config.TEST_SHUFFLE,
            self.model_config.TRAIN_RATIO)
        self.test()
        
        self.classifier_test()
        for epoch in range(1, self.model_config.N_EPOCH + 1):
            self.train(epoch)
            self.test()
            self.classifier_test()
            self.validate()
            
            if self.early_stopper.early_stop:
                self.early_stop_epoch = epoch
                print("early_stop_triggered")
                break
            self.shuffle_loader(trainset, 
            self.model_config.BATCH_SIZE_TRAIN,
            self.model_config.BATCH_SIZE_VAL, 
            self.model_config.TRAIN_RATIO)
        
        print("loading best model to server...")
        self.load()

    
    def run_infernces(self, testsets): 
        self.testloaders = [DataLoader(testdata, batch_size=self.model_config.BATCH_SIZE_TEST, shuffle=False) for testdata in testsets]
        clients_preds, clients_router, server_preds = self.test_inferences()

        return clients_preds, clients_router, server_preds
        
    def pred(self, testset,  pred=True):
        predictions = []
        loader = self._pred_loader(testset, self.model_config.BATCH_SIZE_TEST, self.model_config.TEST_SHUFFLE)
        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*loader)):
                # batches is tuple of batch from each loader
                data_slices = [batch[0].to(self.device).float() for batch in batches]
                target = batches[0][1].to(self.device).long()
                client_activations = []
                for data, client_base_model in zip(data_slices, self.client_base_models):
                    #Client FeedForward
                    client_base_model.eval()
                    activation = client_base_model(data)
                    client_activations.append(activation)
                concat_activations = torch.cat(client_activations, dim=1)
                #Server FeedForward
                self.server_model.eval()
                output = self.server_model(concat_activations)
                pred = output.argmax(dim=1, keepdim=True)
                predictions = predictions + pred.squeeze().tolist()
        return predictions

    def save(self):
        result_path = f"./results/hybrid/{self.model_config.version}/{self.data_config.DATASET_NAME}/{self.seed}"
        os.makedirs(result_path, exist_ok=True)
        model_path = os.path.join(result_path, f'model_server_{self.idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(result_path, f'optimizer_server_{self.idx}.pth').replace("\\", "/")
        torch.save(self.server_model.state_dict(), model_path)
        torch.save(self.SERVER_OPTIMIZER.state_dict(), optimizer_path)

        client_result_path =os.path.join(result_path, "clients")
        os.makedirs(client_result_path, exist_ok=True)
        for i, client_base_model in enumerate(self.client_base_models): 
            model_path = os.path.join(client_result_path, f'model_client_base_{i}.pth').replace("\\", "/")
            optimizer_path = os.path.join(client_result_path, f'optimizer_client_base_{i}.pth').replace("\\", "/")
            torch.save(client_base_model.state_dict(), model_path)
            torch.save(self.CLIENT_BASE_OPTIMIZERS[i].state_dict(), optimizer_path)
                    
        for i, client_classifier_model in enumerate(self.client_classifier_models): 
            model_path = os.path.join(client_result_path, f'model_client_classifier_{i}.pth').replace("\\", "/")
            optimizer_path = os.path.join(client_result_path, f'optimizer_classifier_client_{i}.pth').replace("\\", "/")
            torch.save(client_classifier_model.state_dict(), model_path)
            torch.save(self.CLIENT_CLASSIFIER_OPTIMIZERS[i].state_dict(), optimizer_path)
        
        for i, client_router_model in enumerate(self.client_ig_models): 
            model_path = os.path.join(client_result_path, f'model_client_router_{i}.pth').replace("\\", "/")
            optimizer_path = os.path.join(client_result_path, f'optimizer_router_client_{i}.pth').replace("\\", "/")
            torch.save(client_router_model.state_dict(), model_path)
            torch.save(self.CLIENT_IG_OPTIMIZERS[i].state_dict(), optimizer_path)
                   
    def load(self):
        result_path = f"./results/hybrid/{self.model_config.version}/{self.data_config.DATASET_NAME}/{self.seed}"
        model_path = os.path.join(result_path, f'model_server_{self.idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(result_path, f'optimizer_server_{self.idx}.pth').replace("\\", "/")
        network_state_dict = torch.load(model_path)
        self.server_model.load_state_dict(network_state_dict)
        optimizer_state_dict = torch.load(optimizer_path)
        self.SERVER_OPTIMIZER.load_state_dict(optimizer_state_dict)

        client_result_path = os.path.join(result_path, "clients")
        for i, (client_base_model, model_optimizer) in enumerate(zip(self.client_base_models, self.CLIENT_BASE_OPTIMIZERS)): 
            model_path = os.path.join(client_result_path, f'model_client_base_{i}.pth').replace("\\", "/")
            optimizer_path = os.path.join(client_result_path, f'optimizer_client_base_{i}.pth').replace("\\", "/")
            network_state_dict = torch.load(model_path)
            client_base_model.load_state_dict(network_state_dict)
            optimizer_state_dict = torch.load(optimizer_path)
            model_optimizer.load_state_dict(optimizer_state_dict)
        
        for i, (client_classifier_model, model_optimizer) in enumerate(zip(self.client_classifier_models, self.CLIENT_CLASSIFIER_OPTIMIZERS)): 
            model_path = os.path.join(client_result_path, f'model_client_classifier_{i}.pth').replace("\\", "/")
            optimizer_path = os.path.join(client_result_path, f'optimizer_classifier_client_{i}.pth').replace("\\", "/")
            network_state_dict = torch.load(model_path)
            client_classifier_model.load_state_dict(network_state_dict)
            optimizer_state_dict = torch.load(optimizer_path)
            model_optimizer.load_state_dict(optimizer_state_dict)

        for i, (client_router_model, model_optimizer) in enumerate(zip(self.client_ig_models, self.CLIENT_IG_OPTIMIZERS)): 
            model_path = os.path.join(client_result_path, f'model_client_router_{i}.pth').replace("\\", "/")
            optimizer_path = os.path.join(client_result_path, f'optimizer_router_client_{i}.pth').replace("\\", "/")
            network_state_dict = torch.load(model_path)
            client_router_model.load_state_dict(network_state_dict)
            optimizer_state_dict = torch.load(optimizer_path)
            model_optimizer.load_state_dict(optimizer_state_dict)
                   
                            
    def classifier_test(self):
        test_loss_values = [0 for c in self.client_base_models]
        correct_values = [0 for c in self.client_base_models]
        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*self.testloader)):
                data_slices = [batch[0].to(self.device).float() for batch in batches]
                target = batches[0][1].to(self.device).long()
                for client_idx, client in enumerate(zip(data_slices, self.client_base_models, self.client_classifier_models)):
                    data, client_base_model, classifier_model = client 
                    client_base_model.eval()
                    classifier_model.eval()
                    activation = client_base_model(data)
                    classifier_output = classifier_model(activation)
                    test_loss_values[client_idx] +=  self.LOCAL_CLASSIFIER_CRITERION(classifier_output, target).item()
                    pred = classifier_output.argmax(dim=1, keepdim=True)
                    correct_values[client_idx]+= pred.eq(target.view_as(pred)).sum().item()
        test_loss_values = [t/ len(self.testloader[0].dataset) for t in test_loss_values]
        accuracy_values = [100. * c / len(self.testloader[0].dataset) for c in correct_values]

        if self.log: 
            self.test_losses.append(test_loss_values)
            self.accuracies.append(accuracy_values)
        print(f'\nTest set: Average loss per Sample: {test_loss_values}, Accuracy: {correct_values}/{len(self.testloader[0].dataset)}'
            f'({accuracy_values}%)\n')

    def gate_pred(self): 
        for batch_idx, batches in enumerate(zip(*self.testloader)):
            data_slices = [batch[0].to(self.device).float() for batch in batches]
            target = batches[0][1].to(self.device).long()
            ig_outputs = []
            for client_idx, client in enumerate(zip(data_slices, self.client_base_models, self.client_ig_models)):
                data, client_base_model, client_ig_model = client
                #Client FeedForward
                client_base_model.eval()
                client_ig_model.eval()
                activation = client_base_model(data)
                ig_output = client_ig_model(activation) 
                ig_outputs.append(ig_output)
            return ig_outputs, target
                