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

