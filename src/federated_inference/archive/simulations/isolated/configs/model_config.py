import torch
import torch.optim as optim
import torch.nn as nn

class ModelConfiguration():
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    def __init__(self,  
        model: nn.Module, 
        learning_rate: float = 0.001, 
        batch_size_train: int = 64, 
        batch_size_val: int = 64, 
        batch_size_test: int = 64, 
        train_shuffle: bool = False, 
        val_shuffle: bool = False, 
        test_suffle: bool = False, 
        n_epochs: int = 55,
        train_ratio: float = 0.8):
        self.MODEL = model().to(self.DEVICE)
        self.LEARNING_RATE = learning_rate
        self.TRAIN_RATIO = train_ratio
        self.BATCH_SIZE_TRAIN = batch_size_train
        self.BATCH_SIZE_VAL = batch_size_val
        self.BATCH_SIZE_TEST = batch_size_test
        self.TRAIN_SHUFFLE = train_shuffle
        self.VAL_SHUFFLE = val_shuffle
        self.TEST_SHUFFLE = test_suffle
        self.N_EPOCH = n_epochs
        self.CRITERION = nn.CrossEntropyLoss()
        self.CRITERION_NAME = "CrossEntropyLoss"
        self.OPTIMIZER = optim.Adam(self.MODEL.parameters() , lr=learning_rate) 
        self.OPTIMIZER_NAME = "Adam"

    def set_optimizer(self, optimizer, name):
        self.OPTIMIZER = optimizer
        self.OPTIMIZER_NAME = name

    def set_criterion(self, criterion, name): 
        self.CRITERION = criterion
        self.CRITERION_NAME = name


    def __dict__(self):
        return {
            "model_name": self.MODEL.name,
            "learning_rate": self.LEARNING_RATE, 
            "train_ratio": self.TRAIN_RATIO,
            "batch_size_train": self.BATCH_SIZE_TRAIN,
            "batch_size_val": self.BATCH_SIZE_VAL,
            "batch_size_test": self.BATCH_SIZE_TEST, 
            "train_shuffle": self.TRAIN_SHUFFLE,
            "val_shuffle": self.VAL_SHUFFLE,
            "test_shuffle": self.TEST_SHUFFLE,
            "n_epoch": self.N_EPOCH, 
            "optimizer":  self.OPTIMIZER_NAME, 
            "criterion": self.CRITERION_NAME
        }
