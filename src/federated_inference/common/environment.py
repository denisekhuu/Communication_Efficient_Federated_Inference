from enum import Enum
class LearningMethod(Enum):
    CL = 'Centralized Learning'
    SL = 'Split Learning'
    HFL = 'Horizontal Federated Learning'
    VFL = 'Vertical Federated Learnng'

class DataDistributionType(Enum):
    DIVIDE_EQUALLY = "Divide Equally"

class TransformType(Enum):
    FULL_STRIDE_PARTITION = "Full Stride Partition"

class DataSetEnum(Enum):
    FMNIST = 'FMNIST'
    MNIST = 'MNIST'
    CIFAR10 = 'CIFAR10'

class Member(Enum):
    CLIENT = 'Client'
    SERVER = 'Server'

class DataMode(Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
