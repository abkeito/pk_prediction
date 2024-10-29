# train
import sys
import torch
import random
from torch import optim
from torch import cuda
import matplotlib.pyplot as plt

import data
# import random_data
from model import CoodinatePredictionModel
from pose_prediction import pose_prediction

EPOCH_NUM = 650 # 適宜変えてね
EPOCH_NUM_VAL = 10

train_loss_list = []
test_loss_list = []

# データセットの選択
train_dataset = data.CoodinateData('src/prediction/data/input/train')
test_dataset = data.CoodinateData('src/prediction/data/input/test')
# dataset = random_data.RandomCoodinateData()

input_size = train_dataset.input_dim()
output_size = train_dataset.output_dim()

# 隠れ層のサイズ
hidden_size = 128

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#学習モデルの選択
model = CoodinatePredictionModel(input_size, output_size, hidden_size).to(device)

#最適化方法
lr = 0.02175999695672324 # 学習率
optimizer = optim.Adam(model.parameters(), lr)

inputs = train_dataset.get_inputs()
outputs = train_dataset.get_outputs()

test_inputs = test_dataset.get_inputs()
test_targets = test_dataset.get_outputs()
test_targets = model.padding(test_targets).to(device)

for epoch in range(EPOCH_NUM):
    model.train()
    print("{0} / {1} epoch start.".format(epoch + 1, EPOCH_NUM))

    model.reset_state()
    optimizer.zero_grad()
    loss = model(inputs, outputs)
    loss.backward()
    optimizer.step()
    sum_loss = float(loss.data.to('cpu'))

    # 1フレームあたりのlossを算出
    print("loss = {0}.".format(sum_loss/train_dataset.out_seq_len()))
    train_loss_list.append(sum_loss/train_dataset.out_seq_len())
    
    model_file = "src/prediction/trained_model/prediction_" + str(epoch + 1) + ".model"
    torch.save(model.state_dict(), model_file)

    # test

    model.eval()
    model.reset_state()
    with torch.no_grad():
        test_outputs = model(test_inputs).transpose(0, 1)

    # Test loss の計算
    sum_loss = 0.
    for i, test_output in enumerate(test_outputs):
        loss = model.loss_fn(test_output, test_targets[i])
        sum_loss += loss / test_outputs.size(0) # 系列長でわる
    test_loss = sum_loss / len(test_targets) # バッチ長でわる
    print("test_loss:", test_loss.item())
    test_loss_list.append(test_loss.item())


# 損失の描画
ep = range(1, EPOCH_NUM + 1)
plt.figure()
plt.plot(ep, train_loss_list, label="train")
plt.plot(ep, test_loss_list, label="test")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("src/prediction/data/loss.png")
    
# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_train.py 
    