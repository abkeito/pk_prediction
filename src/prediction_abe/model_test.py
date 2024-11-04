# test
import sys
import torch
import os, json

import data
from data import standardize, destandardize
from model import CoordinatePredictionModel
from pose_prediction import pose_prediction

EPOCH_NUM = 300

# scalerなどを読みとる
with open(os.path.join("/home/u01170/AI_practice/pk_prediction/src/prediction_abe/data_info", "data_and_scaler_info.json"), "r") as f:
    data_info = json.load(f)

mean, std = torch.tensor(data_info["Scaler_Info"]["Mean"]), torch.tensor(data_info["Scaler_Info"]["Scale"])
test_indices = [filename.split('/')[len(filename.split('/'))-1].split('_')[0] for filename in data_info["Dataset_Indices"]["test_indices"]]

for index in test_indices:
    # データセットの選択
    dataset = data.CoordinateData([os.path.join("/home/u01170/AI_practice/pk_prediction/src/making_datasets/optical_flow/dataset", str(index) + "_dataset.json")])

    input_size = dataset.input_dim()
    output_size = dataset.output_dim()

    # デバイスの設定
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # モデルの選択
    model = CoordinatePredictionModel(input_size, output_size).to(device)

    model.load_state_dict(torch.load("src/prediction_abe/trained_model/prediction_{0}.model".format(EPOCH_NUM)))

    inputs = torch.tensor(dataset.get_inputs(), dtype=torch.float32)
    # 標準化
    standardized_inputs = standardize(inputs, mean, std)

    outputs = model(standardized_inputs[0][0])

    #逆標準化
    outputs = destandardize(outputs, mean.to(device), std.to(device))

    # 出力をファイルに保存
    pose_prediction(inputs[0], outputs, os.path.join("/home/u01170/AI_practice/pk_prediction/src/prediction_abe/prediction_data", str(index) + "predict_pose.json"))

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction_abe/model_test.py 
