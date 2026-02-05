from enum import Enum
from federated_inference.common.environment import LearningMethod, DataDistributionType, TransformType
import os
from .transform_config import DataTransformConfiguration

class EnvConfiguration():
    def __init__(self, learning_method: LearningMethod = LearningMethod.CL):
        self.LEARNING_METHOD = learning_method
        self.RESULT_PATH = os.path.join('./results')
    
class FederatedEnvConfiguration(EnvConfiguration):
    def __init__(self, 
        learning_method: LearningMethod = LearningMethod.HFL, 
        data_mode: Data
        n_clients: int = 5, 
        data_distribution_type: DataDistributionType = DataDistributionType.DIVIDE_EQUALLY,
        transform_type: TransformType = TransformType.FULL_STRIDE_PARTITION
        ):
        super().__init__(learning_method)
        if learning_method == LearningMethod.HFL:
            self.data_distribution_type = data_distribution_type
            self.N_CLIENTS = n_clients

        if learning_method == LearningMethod.VFL:
            self.transform_type = transform_type
            self.transform_config = DataTransformConfiguration()
            if transform_type == TransformType.FULL_STRIDE_PARTITION:
                from federated_inference.transform import StridePartitionTransform
                self.transformation = StridePartitionTransform(self.transform_config)
                self.N_CLIENTS = len(self.transformation) 
