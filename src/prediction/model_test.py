# test
import sys
import torch

import data
import model import CodinatePredictionModel
from model_train import EPOCH_NUM

dataset = data.CodinateData() # 引数要確認

input_size = dataset.input_dim()
output_size = dataset.output_dim()

model = CodinatePredictionModel(input_size, output_size)

model.load_state_dict(torch.load("trained_model/prediction_{0}.model".format(EPOCH_NUM)))

output = []