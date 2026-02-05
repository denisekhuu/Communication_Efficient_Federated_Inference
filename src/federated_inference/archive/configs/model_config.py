class ModelConfiguration():
    LEARNING_RATE = 0.001
    BATCH_SIZE_TRAIN = 64
    BATCH_SIZE_VAL = 64
    BATCH_SIZE_TEST = 64
    TRAIN_SHUFFLE = True
    VAL_SHUFFLE = False
    TEST_SHUFFLE = False
    N_EPOCHS = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    def __init__(self, n_epochs):
        self.N_EPOCH = n_epochs

class CLModelConfiguration(ModelConfiguration):
    # percentage of train data to validation data
    TRAIN_RATIO = 0.8

class SLModelConfiguration(ModelConfiguration):
    # percentage of train data to validation data
    TRAIN_RATIO = 0.8

class HFLModelConfiguration(ModelConfiguration):
    # percentage of train data to validation data
    TRAIN_RATIO = 0.8
    N_EPOCHS = 3

class VFLModelConfiguration(ModelConfiguration):
    # percentage of train data to validation data
    TRAIN_RATIO = 0.8
    N_EPOCHS = 3
    TRAIN_SHUFFLE = False