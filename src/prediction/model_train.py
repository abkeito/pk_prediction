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

EPOCH_NUM = 50 # 適宜変えてね
EPOCH_NUM_VAL = 10

train_loss_list = []
val_loss_list = []

# 交差検証
def cross_validate(inputs, outputs, model, k=5):
    print("cross validation start.")

    # データのパディング
    all_inputs = model.padding(inputs).transpose(0, 1)
    all_outputs = model.padding(outputs).transpose(0, 1)

    #データのインデックスをシャッフル
    indices = list(range(len(all_inputs)))
    random.shuffle(indices)

    #インデックスをk個に分割
    fold_size = len(all_inputs) // k
    folds = [indices[i:i + fold_size] for i in range(0, len(indices), fold_size)]

    sum_loss = 0.0
    for i in range(k):
        # インデックスをval, trainそれぞれに指定
        train_indices = [index for j in range(k) if j != i for index in folds[j]]
        val_indices = folds[i]

        # val, trainデータを作成
        train_inputs = all_inputs[train_indices]
        train_outputs = all_outputs[train_indices]
        val_inputs = all_inputs[val_indices]
        val_outputs = all_outputs[val_indices] 

        # 次元の順番を元に戻す
        train_inputs = train_inputs.transpose(0, 1)
        train_outputs = train_outputs.transpose(0, 1)
        val_inputs = val_inputs.transpose(0, 1)
        val_outputs = val_outputs.transpose(0, 1)

        # train
        model.train()
        for epoch in range(EPOCH_NUM_VAL):
            model.reset_state()
            optimizer.zero_grad()
            loss = model(train_inputs, train_outputs)
            loss.backward()
            optimizer.step()

        # validation 
        with torch.no_grad():  # 勾配計算を無効にする
            model.eval()
            val_loss = model(val_inputs, val_outputs)
        sum_loss += val_loss.item()
    print("mean loss = {0}".format(sum_loss/k))
    val_loss_list.append(sum_loss/k)

# データセットの選択
train_dataset = data.CoodinateData('src/prediction/data/input/train')
test_dataset = data.CoodinateData('src/prediction/data/input/test')
# dataset = random_data.RandomCoodinateData()

input_size = train_dataset.input_dim()
output_size = train_dataset.output_dim()

# 隠れ層のサイズ
hidden_size = 50

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#学習モデルの選択
model = CoodinatePredictionModel(input_size, output_size, hidden_size).to(device)

#最適化方法
lr = 0.021028833040228276 # 学習率
optimizer = optim.Adam(model.parameters(), lr)

inputs = train_dataset.get_inputs()
outputs = train_dataset.get_outputs()

model.train()
for epoch in range(EPOCH_NUM):
    print("{0} / {1} epoch start.".format(epoch + 1, EPOCH_NUM))

    model.reset_state()
    optimizer.zero_grad()
    loss = model(inputs, outputs)
    loss.backward()
    optimizer.step()
    sum_loss = float(loss.data.to('cpu'))

    print("loss = {0}.".format(sum_loss))
    train_loss_list.append(sum_loss)
    
    # if (epoch + 1) % 10 == 0:
    model_file = "src/prediction/trained_model/prediction_" + str(epoch + 1) + ".model"
    torch.save(model.state_dict(), model_file)

    cross_validate(inputs, outputs, model, k=5)

# 損失の描画
ep = range(1, EPOCH_NUM + 1)
plt.figure()
# plt.plot(ep, train_loss_list, label="train")
plt.plot(ep, val_loss_list, label="val")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("src/prediction/data/loss_h1000.png")

# test
test = False # テストも行うときはTrueにする
if test:
    # テストデータのリスト
    input_files = test_dataset.get_input_files()

    model.load_state_dict(torch.load("src/prediction/trained_model/prediction_{0}.model".format(60)))
    model.reset_state()

    inputs = test_dataset.get_inputs()

    outputs = model(inputs).transpose(0, 1)

    # 出力をファイルに保存
    pose_prediction(inputs, outputs, input_files)
    
# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_train.py 
    