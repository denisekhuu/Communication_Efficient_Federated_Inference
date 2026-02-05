
from torch.utils.data import Dataset as TorchDataset
from federated_inference.common.cost_calculator import CostCalculator
from federated_inference.common.environment import Member
from collections.abc import Iterable
import logging 
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset
from federated_inference.simulations.naive.configs.model_config import ModelConfiguration
from federated_inference.simulations.naive.configs.data_config import DataConfiguration
from torch.utils.data import Dataset as TorchDataset
from collections.abc import Iterable
import os

class EarlyStopper():
    def __init__(self, patience = 5, min_delta = 0.00001):
        self.patience = patience 
        self.min_delta = min_delta
        self.counter = 0 
        self.best_loss = None 
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None: 
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else: 
            self.best_loss = val_loss
            self.counter = 0
        

class NaiveVerticalServer():

    def __init__(self, 
            idx, 
            seed,
            model_config: ModelConfiguration,
            data_config: DataConfiguration,
            log: bool = True, 
            log_interval: int = 100,
            save_interval: int = 20
        ):
        self.idx = idx
        self.seed = seed
        self.model_config = model_config
        self.data_config = data_config
        self.seed = seed
        self.n_epoch = model_config.N_EPOCH
        self.device = model_config.DEVICE
        self.member_type = Member.SERVER
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

    def combine_batch_columns(self, batch_list, expect_target = True):
        """
        batch_list: list of batches from each dataset loader
        Each batch is usually a tuple (inputs, targets)
        """
        combined_targets = None
        combined_inputs = None
        if expect_target:        
            inputs_list = [b[0] for b in batch_list]  # get inputs from each batch
            combined_inputs = torch.cat(inputs_list, dim=3)
            targets_list = [b[1] for b in batch_list] 
            combined_targets = targets_list[0] 
        else:
            inputs_list = batch_list
            combined_inputs = torch.cat(inputs_list, dim=3)
        return combined_inputs, combined_targets

    def _to_loader(self, trainsets, testsets, batch_size_train, batch_size_val, batch_size_test, train_shuffle, val_shuffle, test_shuffle, train_ratio):
        # TODO refactoing to use self
        if train_shuffle:
            # Assuming all trainsets have the same length
            dataset_length = len(trainsets[0])
            assert all(len(trainset) == dataset_length for trainset in trainsets), "All trainsets must be the same length"

            indices = np.arange(dataset_length)

            if train_shuffle:
                np.random.shuffle(indices)

            train_end = round(train_ratio * dataset_length)
            train_indices = indices[:train_end]
            val_indices = indices[train_end:]

            traindatas = [Subset(trainset, train_indices) for trainset in trainsets]
            valdatas = [Subset(trainset, val_indices) for trainset in trainsets]
        else:
            traindatas = [Subset(trainset, range(round(train_ratio*len(trainset)))) for trainset in trainsets]
            valdatas = [Subset(trainset, range(round(train_ratio*len(trainset)), len(trainset))) for trainset in trainsets]
        self.trainloader = [DataLoader(traindata, batch_size=batch_size_train, shuffle=False)  for traindata in traindatas]
        self.valloader = [DataLoader(valdata, batch_size=batch_size_val, shuffle=False) for valdata in valdatas]
        self.testloader = [DataLoader(testdata, batch_size=batch_size_test, shuffle=False) for testdata in testsets]


    def _pred_loader(self, testsets, batch_size_test, test_shuffle):
        return  [DataLoader(testdata, batch_size=batch_size_test, shuffle=False) for testdata in testsets]

    def train(self, epoch):
        for batch_idx, batches in enumerate(zip(*self.trainloader)):
            # batches is tuple of batch from each loader
            data, target = self.combine_batch_columns(batches, expect_target=True)
            data = data.to(self.device).float()
            target = target.to(self.device).long()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if self.log and batch_idx % self.log_interval == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.trainloader[0].dataset)} '
                    f'({100. * batch_idx / len(self.trainloader[0]):.0f}%)]\tLoss: {loss.item():.6f}')

            if batch_idx % self.save_interval == 0:
                self.train_losses.append(loss.item())
        val_loss = 0
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*self.valloader)):
                data, target = self.combine_batch_columns(batches, expect_target=True)
                data = data.to(self.device).float()
                target = target.to(self.device).long()
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
        val_loss /= len(self.valloader[0].dataset)
    
        if self.early_stopper.best_loss is None or val_loss < self.early_stopper.best_loss:
            print("Validation loss improved. Saving model...")
            self.save()
        self.early_stopper(val_loss)

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*self.testloader)):
                # batches is tuple of batch from each loader
                data, target = self.combine_batch_columns(batches)
                data = data.to(self.device).float()
                target = target.to(self.device).long()
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.testloader[0].dataset)
        accuracy = 100. * correct / len(self.testloader[0].dataset)
        if self.log: 
            self.test_losses.append(test_loss)
            self.accuracies.append(accuracy)
        print(f'\nTest set: Average loss per sample: {test_loss:.4f}, Accuracy: {correct}/{len(self.testloader[0].dataset)} '
            f'({accuracy:.0f}%)\n')
            
    def run_training(self, trainset, testset):
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

    def pred(self, testset):
        self.model.eval()
        predictions = []
        loader = self._pred_loader(testset, self.model_config.BATCH_SIZE_TEST, self.model_config.TEST_SHUFFLE)
        with torch.no_grad():
            for batch_idx, batches in enumerate(zip(*loader)):
                data, _ = self.combine_batch_columns(batches)
                data = data.to(self.device).float()
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                predictions = predictions + pred.squeeze().tolist()
        cost = CostCalculator.calculate_communication_cost_by(predictions)
        cost.set_cost_reason("send_predictions")
        self.costs.append(cost)
        return predictions

    def save(self):
        result_path = f"./results/naive/{self.data_config.DATASET_NAME}/{self.seed}"
        os.makedirs(result_path, exist_ok=True)
        model_path = os.path.join(result_path, f'model_server_{self.idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(result_path, f'optimizer_server_{self.idx}.pth').replace("\\", "/")
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)


    def load(self):
        result_path = f"./results/naive/{self.data_config.DATASET_NAME}/{self.seed}"
        model_path = os.path.join(result_path, f'model_server_{self.idx}.pth').replace("\\", "/")
        optimizer_path = os.path.join(result_path, f'optimizer_server_{self.idx}.pth').replace("\\", "/")
        network_state_dict = torch.load(model_path)
        self.model.load_state_dict(network_state_dict)
        optimizer_state_dict = torch.load(optimizer_path)
        self.optimizer.load_state_dict(optimizer_state_dict)