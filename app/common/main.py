from federated_inference.dataset import MNISTDataset, FMNISTDataset
from federated_inference.configs.data_config import DataConfiguration
from federated_inference.transform import StridePartitionTransform, DataSplitter, SensorWrapper
from torch.utils.data import DataLoader

class DataTransformConfiguration(): 

    def __init__(self, tensor_size = [1,28,28], method_name = 'full', mask_size = [14,14], dimensions = [1,2], stride = 14, n_position = None, drop_p = None):
        self.TENSOR_SIZE = tensor_size
        self.METHOD_NAME = method_name
        self.MASK_SIZE = mask_size 
        self.DIMENSIONS = dimensions
        self.STRIDE = stride
        self.N_POSITION = n_position
        self.DROP_P = drop_p

    def __dict__(self):
        return {
            "tensor_size": self.TENSOR_SIZE, 
            "method_name": self.METHOD_NAME,
            "mask_size": self.MASK_SIZE,
            "dimensions": self.DIMENSIONS,
            "stride": self.STRIDE,
            "n_position": self.N_POSITION,
            "drop_p": self.DROP_P,

        }
    
class Client():
    def __init__(self, idx, trainset, testset):
        self.id = idx
        self.trainset = trainset
        self.testset = testset

configs = DataConfiguration()
##___________ Data Page ___________ ##
data = FMNISTDataset(configs)
trainloader = DataLoader(data.train_dataset, batch_size=9, shuffle=True)    

##___________ Clients Page ___________ ##
data_transform_config = DataTransformConfiguration()
op_transform = StridePartitionTransform(data_transform_config)
number_of_clients = len(op_transform)
transformed_data = FMNISTDataset(configs, [op_transform])
grouped_trainset = DataSplitter.group_dataset(transformed_data.train_dataset)
grouped_testset = DataSplitter.group_dataset(transformed_data.test_dataset)
grouped_sets = {k: (grouped_trainset[k], grouped_testset[k]) for k in sorted(grouped_trainset)}

##___________ Client Page ___________ ##

clients = [
    Client(i, 
        SensorWrapper(transformed_data.train_dataset, i), 
        SensorWrapper(transformed_data.test_dataset, i)) 
    for i in range(number_of_clients)]