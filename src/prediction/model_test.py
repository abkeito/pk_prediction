# test
import sys
import torch

import data
from model import CoodinatePredictionModel
from pose_prediction import pose_prediction

# データセットの選択
dataset = data.CoodinateData()

input_size = dataset.input_dim()
output_size = dataset.output_dim()

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# モデルの選択
model = CoodinatePredictionModel(input_size, output_size).to(device)

model.load_state_dict(torch.load("src/prediction/trained_model/prediction_{0}.model".format(1000)))

inputs = dataset.get_inputs()

outputs = model(inputs).transpose(0, 1)

# 出力をファイルに保存
pose_prediction(inputs, outputs)

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_test.py 
