# test
import sys
import torch
import os 

import data
from data import standardize, destandardize
from model import CoordinatePredictionModel
from pose_prediction import pose_prediction

EPOCH_NUM = 1000

# データセットの選択
dataset = data.CoordinateData([os.path.join("data", "6.mp4_pose.json")])

input_size = dataset.input_dim()
output_size = dataset.output_dim()

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# モデルの選択
model = CoordinatePredictionModel(input_size, output_size).to(device)

model.load_state_dict(torch.load("src/prediction/trained_model/prediction_{0}.model".format(EPOCH_NUM)))

inputs = torch.tensor(dataset.get_inputs(), dtype=torch.float32)
# 標準化
standardized_inputs = standardize(inputs)

outputs = model(standardized_inputs[0][0])

#逆標準化
outputs = destandardize(outputs, standardized_inputs[1].to(device), standardized_inputs[2].to(device))

# 出力をファイルに保存
pose_prediction(inputs[0], outputs, os.path.join("data", "predict_pose.json"))

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_test.py 
