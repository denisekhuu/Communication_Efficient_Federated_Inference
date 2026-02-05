
import torch
import torch.nn as nn

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 64
BATCH_SIZE_VAL = 64

TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TEST_SHUFFLE = False

N_EPOCHS = 50
TRAIN_RATIO = 0.8

class HybridVFLWoRouterModelConfiguration():
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    def __init__(self,  
        version: str,
        server_model: nn.Module, 
        client_base_model: nn.Module, 
        client_classifier_head: nn.Module,       
        learning_rate: float = LEARNING_RATE, 
        batch_size_train: int = BATCH_SIZE_TRAIN, 
        batch_size_val: int = BATCH_SIZE_TEST, 
        batch_size_test: int = BATCH_SIZE_VAL, 
        train_shuffle: bool = TRAIN_SHUFFLE, 
        val_shuffle: bool = VAL_SHUFFLE, 
        test_suffle: bool = TEST_SHUFFLE, 
        n_epochs: int = N_EPOCHS,
        train_ratio: float = TRAIN_RATIO):
        self.version = version
        self.SERVER_MODEL = server_model
        self.CLIENT_BASE_MODEL = client_base_model
        self.CLIENT_CLASSIFIER_MODEL = client_classifier_head
        self.LEARNING_RATE = learning_rate
        self.TRAIN_RATIO = train_ratio
        self.BATCH_SIZE_TRAIN = batch_size_train
        self.BATCH_SIZE_VAL = batch_size_val
        self.BATCH_SIZE_TEST = batch_size_test
        self.TRAIN_SHUFFLE = train_shuffle
        self.VAL_SHUFFLE = val_shuffle
        self.TEST_SHUFFLE = test_suffle
        self.N_EPOCH = n_epochs

    def __dict__(self):
        return {
            "learning_rate": self.LEARNING_RATE, 
            "train_ratio": self.TRAIN_RATIO,
            "batch_size_train": self.BATCH_SIZE_TRAIN,
            "batch_size_val": self.BATCH_SIZE_VAL,
            "batch_size_test": self.BATCH_SIZE_TEST, 
            
            "train_shuffle": self.TRAIN_SHUFFLE,
            "val_shuffle": self.VAL_SHUFFLE,
            "test_shuffle": self.TEST_SHUFFLE,
            "n_epoch": self.N_EPOCH, 
        }



class HybridSplitModelConfiguration():
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    def __init__(self,  
        version: str,
        server_model: nn.Module, 
        client_base_model: nn.Module, 
        client_classifier_head: nn.Module, 
        client_ig_head: nn.Module,          
        learning_rate: float = LEARNING_RATE, 
        batch_size_train: int = BATCH_SIZE_TRAIN, 
        batch_size_val: int = BATCH_SIZE_TEST, 
        batch_size_test: int = BATCH_SIZE_VAL, 
        train_shuffle: bool = TRAIN_SHUFFLE, 
        val_shuffle: bool = VAL_SHUFFLE, 
        test_suffle: bool = TEST_SHUFFLE, 
        n_epochs: int = N_EPOCHS,
        train_ratio: float = TRAIN_RATIO):
        self.version = version
        self.SERVER_MODEL = server_model
        self.CLIENT_BASE_MODEL = client_base_model
        self.CLIENT_CLASSIFIER_MODEL = client_classifier_head
        self.CLIENT_IG_MODEL = client_ig_head
        self.LEARNING_RATE = learning_rate
        self.TRAIN_RATIO = train_ratio
        self.BATCH_SIZE_TRAIN = batch_size_train
        self.BATCH_SIZE_VAL = batch_size_val
        self.BATCH_SIZE_TEST = batch_size_test
        self.TRAIN_SHUFFLE = train_shuffle
        self.VAL_SHUFFLE = val_shuffle
        self.TEST_SHUFFLE = test_suffle
        self.N_EPOCH = n_epochs

    def __dict__(self):
        return {
            "learning_rate": self.LEARNING_RATE, 
            "train_ratio": self.TRAIN_RATIO,
            "batch_size_train": self.BATCH_SIZE_TRAIN,
            "batch_size_val": self.BATCH_SIZE_VAL,
            "batch_size_test": self.BATCH_SIZE_TEST, 
            
            "train_shuffle": self.TRAIN_SHUFFLE,
            "val_shuffle": self.VAL_SHUFFLE,
            "test_shuffle": self.TEST_SHUFFLE,
            "n_epoch": self.N_EPOCH, 
        }


import torch.optim as optim

class OnDeviceModelConfiguration():
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    def __init__(self,  
        version: str,
        model: nn.Module, 
        learning_rate: float = LEARNING_RATE, 
        batch_size_train: int = BATCH_SIZE_TRAIN, 
        batch_size_val: int = BATCH_SIZE_VAL, 
        batch_size_test: int = BATCH_SIZE_TEST, 
        train_shuffle: bool = TRAIN_SHUFFLE, 
        val_shuffle: bool = VAL_SHUFFLE, 
        test_suffle: bool = TEST_SHUFFLE, 
        n_epochs: int = N_EPOCHS,
        train_ratio: float = TRAIN_RATIO):
        self.version = version
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
        self.OPTIMIZER = optim.Adam(self.MODEL.parameters() , lr=learning_rate, weight_decay=WEIGHT_DECAY) 
        self.OPTIMIZER_NAME = "Adam"


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


import torch
import torch.optim as optim
import torch.nn as nn

class OnCloudModelConfiguration():
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    def __init__(self,  
        seed: int, 
        version: str,
        model: nn.Module, 
        learning_rate: float = LEARNING_RATE, 
        batch_size_train: int = BATCH_SIZE_TRAIN, 
        batch_size_val: int = BATCH_SIZE_VAL, 
        batch_size_test: int = BATCH_SIZE_TEST, 
        train_shuffle: bool = TRAIN_SHUFFLE, 
        val_shuffle: bool = VAL_SHUFFLE,
        test_suffle: bool = TEST_SHUFFLE, 
        n_epochs: int = N_EPOCHS,
        train_ratio: float = TRAIN_RATIO):
        self.seed = seed
        self.version = version
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
        self.OPTIMIZER = optim.Adam(self.MODEL.parameters() , lr=learning_rate, weight_decay=WEIGHT_DECAY) 
        self.OPTIMIZER_NAME = "Adam"

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



import torch
import torch.nn as nn
import torch.optim as optim

class OnCloudCIFAR10ModelConfiguration:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    def __init__(
        self,  
        seed: int, 
        version: str,
        model: nn.Module, 
        learning_rate: float = LEARNING_RATE, 
        batch_size_train: int = BATCH_SIZE_TRAIN, 
        batch_size_val: int = BATCH_SIZE_VAL, 
        batch_size_test: int = BATCH_SIZE_TEST, 
        train_shuffle: bool = TRAIN_SHUFFLE, 
        val_shuffle: bool = VAL_SHUFFLE,
        test_suffle: bool = TEST_SHUFFLE, 
        n_epochs: int = N_EPOCHS,
        train_ratio: float = TRAIN_RATIO,
        momentum: float = 0.9  # for SGD
    ):
        self.seed = seed
        self.version = version
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

        self.OPTIMIZER = optim.SGD(
            self.MODEL.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=WEIGHT_DECAY
        )
        self.OPTIMIZER_NAME = f"SGD (momentum={momentum})"

    def as_dict(self):
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