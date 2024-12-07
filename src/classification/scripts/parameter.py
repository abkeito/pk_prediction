import torch
from torch import cuda

# 学習時のデバイス、損失関数、最適化方法を持っておくクラス
class Train_parameter():
    def __init__(self, criterion=None, optimizer=None):
        if cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.criterion = criterion
        self.optimizer = optimizer