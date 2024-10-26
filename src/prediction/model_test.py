# test
import sys
import torch

import data
from data import standardize, destandardize
from model import CoodinatePredictionModel
from pose_prediction import pose_prediction

EPOCH_NUM = 10

# データセットの選択
dataset = data.CoodinateData("src/prediction/data/pose.json")

input_size = dataset.input_dim()
output_size = dataset.output_dim()

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# モデルの選択
model = CoodinatePredictionModel(input_size, output_size).to(device)

model.load_state_dict(torch.load("src/prediction/trained_model/prediction_{0}.model".format(EPOCH_NUM)))

inputs = torch.tensor(dataset.get_inputs(), dtype=torch.float32)
# 標準化
standardized_inputs = standardize(inputs)

outputs = model(standardized_inputs[0])

#逆標準化
outputs = destandardize(outputs, standardized_inputs[1].to(device), standardized_inputs[2].to(device))

# 出力をファイルに保存
pose_prediction(inputs, outputs, "src/prediction/data/predicted_pose.json")

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_test.py 
