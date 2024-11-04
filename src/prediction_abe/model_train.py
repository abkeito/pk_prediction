# train
import sys
import torch
import random
from torch import optim
from torch import cuda
import os, json
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import data
from data import standardize
# import random_data
from model import CoordinatePredictionModel

EPOCH_NUM = 300 # 適宜変えてね

# wandb を設定
wandb.init(
    # プロジェクト名を設定
    project="PK-project", 
    # 実行名をつける(これまでのように、指定しない場合は、自動で作成してくれます）
    # ハイパーパラメータ等の情報を保存
    config={
    "learning_rate": 1,
    "architecture": "LSTM",
    "dataset": "dataset_v2",
    "epochs": 100,
    })
# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(torch.version.cuda)
print(torch.__version__)
device_availability = 'gpu' if torch.cuda.is_available() else 'cpu'
print(device_availability)

# データセットの選択
folder = "/home/u01170/AI_practice/pk_prediction/src/making_datasets/optical_flow/dataset/"
filepaths = [os.path.join(folder, filepath) for filepath in os.listdir(folder)]
dataset = data.CoordinateData(filepaths)
input_size, output_size = dataset.input_dim(), dataset.output_dim()
batch_size, node_size = dataset.batch_size, dataset.node_size*2
input_frame_size, output_frame_size = dataset.input_seqsize, dataset.output_seqsize

scaler = StandardScaler()

# 標準化 -> (batch_size * frame_size * node_size) をreshape したり、input と output を合わせて標準化
#print(np.array(dataset.get_inputs()).shape)
input, output = torch.tensor(dataset.get_inputs(), dtype = torch.float32), torch.tensor(dataset.get_outputs(), dtype = torch.float32)
combined = torch.cat((input.reshape(-1, node_size), output.reshape(-1, node_size)), dim=0)
scaler.fit(combined)
input, output = combined[:batch_size*input_frame_size,:].reshape(batch_size, input_frame_size, node_size), combined[batch_size*output_frame_size:,:].reshape(batch_size, input_frame_size, node_size)

print("scaler info ", scaler.mean_, scaler.var_)
print("shape of the data: input, output = ", input.shape, output.shape)

#データの分割
indices = np.array(range(input.shape[0]))
train_valid_input, test_input, train_valid_output, test_output, train_valid_indices, test_indices  = train_test_split(input, output, indices, train_size=0.9)
train_input, valid_input, train_output, valid_output = train_test_split(train_valid_input, train_valid_output, train_size=0.9)
print("train: ", len(train_input), len(train_output))
print("valid: ",len(valid_input), len(valid_output))
print("test: ",len(test_input), len(test_output), test_indices)

# meta情報保存
# 保存するデータを辞書形式でまとめる
data_info = {
    "Dataset_Indices": {
        "train_valid_indices": [filepaths[idx] for idx in train_valid_indices.tolist()],
        "test_indices": [filepaths[idx] for idx in test_indices.tolist()]
    },
    "Scaler_Info": {
        "Mean": scaler.mean_.tolist(),
        "Scale": scaler.scale_.tolist()
    },
}

# JSONファイルに書き込み
with open(os.path.join("/home/u01170/AI_practice/pk_prediction/src/prediction_abe/data_info", "data_and_scaler_info.json"), "w") as file:
    json.dump(data_info, file, indent=4)

#学習モデルの選択
model = CoordinatePredictionModel(input_size, output_size).to(device)

#最適化方法
optimizer = optim.Adam(model.parameters())

# 学習
for epoch in range(EPOCH_NUM):
    print("{0} / {1} epoch start.".format(epoch + 1, EPOCH_NUM))

    sum_loss = 0
    for i, (input, output) in enumerate(zip(train_input, train_output)):
        model.reset_state()
        optimizer.zero_grad()

        loss = model(input, output)
        loss.backward()
        optimizer.step()
        sum_loss += float(loss.data.to('cpu'))

    train_loss = sum_loss / len(train_input)
    print("mean loss = {0}.".format(train_loss))
    wandb.log({"train_loss": train_loss})
    # cross_validate(dataset, k=5)

    # エポック毎に書き出す
    if (epoch+1) % 10 == 0:
        sum_valid_loss = 0
        for i, (input, output) in enumerate(zip(valid_input, valid_output)):
            model.reset_state()
            valid_loss = model(input, output)
            sum_valid_loss += float(valid_loss.data.to('cpu'))
        valid_loss = sum_valid_loss / len(valid_input)
        print("Validation Loss: ", valid_loss)
        wandb.log({"valid_loss": valid_loss})
        model_file = "src/prediction_abe/trained_model/prediction_" + str(epoch + 1) + ".model"
        torch.save(model.state_dict(), model_file)

wandb.finish()
# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction_abe/model_train.py 