# train
import sys
import torch
import random
from torch import optim
from torch import cuda
<<<<<<< HEAD
import os 
=======
import matplotlib.pyplot as plt
>>>>>>> main

import data
# import random_data
<<<<<<< HEAD
from model import CoordinatePredictionModel

EPOCH_NUM = 1000 # 適宜変えてね

# 交差検証
def cross_validate(dataset, k=5):
    # テンソル化してbatchsizeとseqsizeの次元を入れ替え
    all_inputs = torch.tensor(dataset.get_inputs(), dtype=torch.float32).transpose(0, 1)
    all_outputs = torch.tensor(dataset.get_outputs(), dtype=torch.float32).transpose(0, 1)

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
        for epoch in range(EPOCH_NUM):
            model.reset_state()
            optimizer.zero_grad()
            loss = model(train_inputs, train_outputs)
            loss.backward()
            optimizer.step()

        # validation 
        model.reset_state()
        val_loss = model(val_inputs, val_outputs)
        sum_loss += val_loss.item()
    print("mean loss = {0}".format(sum_loss/k))

# データセットの選択
filepaths = [os.path.join(("data"), filepath) for filepath in os.listdir("data")]
dataset = data.CoordinateData(filepaths)
# dataset = random_data.RandomCoordinateData()
=======
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
>>>>>>> main

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
<<<<<<< HEAD
model = CoordinatePredictionModel(input_size, output_size).to(device)
=======
model = CoodinatePredictionModel(input_size, output_size, hidden_size).to(device)
>>>>>>> main

#最適化方法
lr = 0.02175999695672324 # 学習率
optimizer = optim.Adam(model.parameters(), lr)

inputs = train_dataset.get_inputs()
outputs = train_dataset.get_outputs()

test_inputs = test_dataset.get_inputs()
test_targets = test_dataset.get_outputs()
test_targets = model.padding(test_targets).to(device)

<<<<<<< HEAD
# データセット準備
inputs = torch.tensor(dataset.get_inputs(), dtype = torch.float32)
outputs = torch.tensor(dataset.get_outputs(), dtype = torch.float32)

#標準化
standardized_inputs = standardize(inputs)
standardized_outputs = standardize(outputs, standardized_inputs[1], standardized_inputs[2])

# 学習
=======
>>>>>>> main
for epoch in range(EPOCH_NUM):
    model.train()
    print("{0} / {1} epoch start.".format(epoch + 1, EPOCH_NUM))

<<<<<<< HEAD
    sum_loss = 0
    for i, (input, output) in enumerate(zip(standardized_inputs[0], standardized_outputs[0])):
        model.reset_state()
        optimizer.zero_grad()

        loss = model(input, output)
        loss.backward()
        optimizer.step()
        sum_loss += float(loss.data.to('cpu'))

    print("mean loss = {0}.".format(sum_loss / input_size))
    # cross_validate(dataset, k=5)

    # エポック毎に書き出す
    if (epoch+1) % 100 == 0:
        model_file = "src/prediction/trained_model/prediction_" + str(epoch + 1) + ".model"
        torch.save(model.state_dict(), model_file)

=======
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
    
>>>>>>> main
# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_train.py 
    