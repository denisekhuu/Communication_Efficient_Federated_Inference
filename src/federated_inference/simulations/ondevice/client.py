import os
from collections.abc import Iterable

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset as TorchDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score,  confusion_matrix

from federated_inference.common.environment import Member
from federated_inference.common.early_stopper import EarlyStopper


class OnDeviceVerticalClient():

    def __init__(self, 
            idx, 
            seed,
            model_config,
            data_config,
            dataset: TorchDataset,
            labels,
            log: bool = True, 
            log_interval: int = 100,
            save_interval: int = 20
        ):
        self.idx = idx
        self.seed = seed
        self.data = dataset
        self.data_config = data_config
        self.labels = labels
        self.numerical_labels = range(len(labels))
        self.member_type = Member.CLIENT
        self.model_config = model_config
        self.n_epoch = model_config.N_EPOCH
        self.device = model_config.DEVICE
        self.model = model_config.MODEL
        self.optimizer = model_config.OPTIMIZER
        self.criterion = model_config.CRITERION
        self.costs = []

        self.log = log
        self.log_interval = log_interval
        self.save_interval = save_interval

        if self.log: 
            self.train_losses = []
            self.test_losses = []
            self.accuracies = []

    def select_subset(self, ids: Iterable[int], set_type: str = "train"):
        if set_type == "test":
            return Subset(self.data.test_dataset, ids)
        else: 
            return Subset(self.data.train_dataset, ids)


    def _to_loader(self, trainset, testset, batch_size_train, batch_size_val, batch_size_test, train_shuffle, val_shuffle, test_shuffle, train_ratio):
        self.train_set_indices = np.arange(len(trainset))
        traindata = Subset(trainset, range(round(train_ratio*len(trainset))))
        valdata = Subset(trainset, range(round(train_ratio*len(trainset)), len(trainset)))
        self.trainloader = DataLoader(traindata, batch_size=batch_size_train, shuffle= train_shuffle) 
        self.valloader = DataLoader(valdata, batch_size=batch_size_val, shuffle= val_shuffle) 
        self.testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=test_shuffle)

    def shuffle_loader(self, trainset, batch_size_train, batch_size_val, train_ratio):
        np.random.shuffle(self.train_set_indices)
        dataset_length = len(self.train_set_indices)
        train_end = round(train_ratio * dataset_length)
        train_indices = self.train_set_indices[:train_end]
        val_indices = self.train_set_indices[train_end:]
        traindata = Subset(trainset, train_indices)
        valdata = Subset(trainset, val_indices)
        self.valloader = DataLoader(valdata, batch_size=batch_size_val, shuffle=False) 
        self.trainloader = DataLoader(traindata, batch_size=batch_size_train, shuffle=False) 

    def _pred_loader(self, testset, batch_size_test, test_shuffle):
        return DataLoader(testset, batch_size=batch_size_test, shuffle=test_shuffle)

    def train(self, epoch):
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            data = data.to(self.model_config.DEVICE).float()
            target = target.to(self.model_config.DEVICE).long()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.save_interval == 0: 
                self.train_losses.append(loss.item())
            if self.log and batch_idx % self.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.trainloader.dataset)} '
                    f'({100. * batch_idx / len(self.trainloader):.0f}%)]\tLoss: {loss.item():.6f}')


        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for data, target in self.valloader:
                data = data.to(self.model_config.DEVICE).float()
                target = target.to(self.model_config.DEVICE).long()
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
        val_loss /= len(self.valloader.dataset)

        if self.early_stopper.best_loss is None or val_loss < self.early_stopper.best_loss:
            print("Validation loss improved. Saving model...")
            self.save()
        self.early_stopper(val_loss)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.testloader:
                data = data.to(self.model_config.DEVICE).float()
                target = target.to(self.model_config.DEVICE).long()
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.testloader.dataset)
        accuracy = 100. * correct / len(self.testloader.dataset)
        if self.log: 
            self.test_losses.append(test_loss)
            self.accuracies.append(accuracy)
        print(f'\nTest set: Average loss per sample: {test_loss:.4f}, Accuracy: {correct}/{len(self.testloader.dataset)} '
            f'({accuracy:.0f}%)\n')

    def run_training(self):
        trainset = self.data.train_dataset
        testset = self.data.test_dataset
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
        for epoch in range(1, self.model_config.N_EPOCH + 1):
            self.train(epoch)
            self.test()
            if self.early_stopper.early_stop:
                self.early_stop_epoch = epoch
                print("early_stop_triggered")
                break
            self.shuffle_loader(trainset, 
            self.model_config.BATCH_SIZE_TRAIN,
            self.model_config.BATCH_SIZE_VAL, 
            self.model_config.TRAIN_RATIO)
                
            
    def save(self):
        result_path = f"./results/ondevice/{self.model_config.version}/{self.data_config.DATASET_NAME}/{self.seed}"
        os.makedirs(result_path, exist_ok=True)
        model_path = os.path.join(result_path, f'model_client_{self.idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(result_path, f'optimizer_client_{self.idx}.pth').replace("\\", "/")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)


    def load(self):
        result_path = f"./results/ondevice/{self.model_config.version}/{self.data_config.DATASET_NAME}/{self.seed}"
        model_path = os.path.join(result_path, f'model_client_{self.idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(result_path, f'optimizer_client_{self.idx}.pth').replace("\\", "/")
        network_state_dict = torch.load(model_path)
        self.model.load_state_dict(network_state_dict)
        optimizer_state_dict = torch.load(optimizer_path)
        self.optimizer.load_state_dict(optimizer_state_dict)

    def pred(self):
        predictions = []
        testset = self.data.test_dataset
        testloader = self._pred_loader(testset, self.model_config.BATCH_SIZE_TEST, self.model_config.TEST_SHUFFLE)
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in testloader :
                data = data.to(self.model_config.DEVICE).float()
                target = target.to(self.model_config.DEVICE).long()
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                predictions = predictions + pred.squeeze().tolist()

        return predictions
    
    def _pred(self, testset): 
        testloader = self._pred_loader(testset, self.model_config.BATCH_SIZE_TEST, self.model_config.TEST_SHUFFLE)
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for data, target in testloader :
                data = data.to(self.model_config.DEVICE).float()
                target = target.to(self.model_config.DEVICE).long()
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                predictions += pred.squeeze().tolist()

        return predictions

    def check(self, predicted_labels,  pred_all: bool= True):

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