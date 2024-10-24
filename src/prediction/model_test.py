# test
import sys
import torch

import data
from model import CoodinatePredictionModel
EPOCH_NUM = 10

dataset = data.CoodinateData("data/pose.json")

input_size = dataset.input_dim()
output_size = dataset.output_dim()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CoodinatePredictionModel(input_size, output_size).to(device)

model.load_state_dict(torch.load("trained_model/prediction_{0}.model".format(EPOCH_NUM)))

inputs = torch.tensor(dataset.get_inputs(), dtype=torch.float32)
outputs = model(inputs)
print(outputs[:,0])

# 出力をどう扱うかは未定