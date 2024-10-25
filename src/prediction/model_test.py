# test
import sys
import torch

import data
from data import standardize, destandardize
from model import CoodinatePredictionModel
EPOCH_NUM = 10

# データセットの選択
dataset = data.CoodinateData("data/pose.json")

input_size = dataset.input_dim()
output_size = dataset.output_dim()

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# モデルの選択
model = CoodinatePredictionModel(input_size, output_size).to(device)

model.load_state_dict(torch.load("trained_model/prediction_{0}.model".format(EPOCH_NUM)))

inputs = torch.tensor(dataset.get_inputs(), dtype=torch.float32)
# 標準化
standardized_inputs = standardize(inputs)

outputs = model(standardized_inputs[0])

#逆標準化
outputs = destandardize(outputs, standardized_inputs[1].to(device), standardized_inputs[2].to(device))

# 出力をファイルに保存
