# test
import sys
import torch
import os 

import data
<<<<<<< HEAD
from data import standardize, destandardize
from model import CoordinatePredictionModel
from pose_prediction import pose_prediction

EPOCH_NUM = 1000

# データセットの選択
dataset = data.CoordinateData([os.path.join("data", "6.mp4_pose.json")])
=======
from model import CoodinatePredictionModel
from pose_prediction import pose_prediction

MODEL_NUM = 510 # 使うモデルの番号

# データセットの選択
test_dataset = data.CoodinateData('src/prediction/data/input/test')
>>>>>>> main

input_size = test_dataset.input_dim()
output_size = test_dataset.output_dim()

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

hidden_size = 128 # trainと合わせる

# モデルの選択
<<<<<<< HEAD
model = CoordinatePredictionModel(input_size, output_size).to(device)
=======
model = CoodinatePredictionModel(input_size, output_size, hidden_size).to(device)
>>>>>>> main

# テストデータのリスト
input_files = test_dataset.get_input_files()

model.load_state_dict(torch.load("src/prediction/trained_model/prediction_{0}.model".format(MODEL_NUM)))

<<<<<<< HEAD
outputs = model(standardized_inputs[0][0])
=======
inputs = test_dataset.get_inputs()
targets = test_dataset.get_outputs()
targets = model.padding(targets).to(device)
>>>>>>> main

model.eval()
with torch.no_grad():
    outputs = model(inputs).transpose(0, 1)

# 出力をファイルに保存
<<<<<<< HEAD
pose_prediction(inputs[0], outputs, os.path.join("data", "predict_pose.json"))
=======
pose_prediction(inputs, outputs, input_files)

# Test loss の計算
sum_loss = 0.
for i, output in enumerate(outputs):
    loss = model.loss_fn(output, targets[i])
    sum_loss += loss / outputs.size(0) # 系列長でわる
test_loss = sum_loss / len(targets) # バッチ長でわる
print("test_loss:", test_loss.item())
>>>>>>> main

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_test.py 
