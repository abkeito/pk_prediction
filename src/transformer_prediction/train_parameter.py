import torch
from torch import cuda

class Train_parameter():
    def __init__(self, criterion=None, optimizer=None):
        if cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.criterion = criterion
        self.optimizer = optimizer