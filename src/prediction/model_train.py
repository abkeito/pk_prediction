# train
import sys
import torch
import random
from torch import optim
from torch import cuda

import data
# import random_data
from model import CoodinatePredictionModel

EPOCH_NUM = 100 # 適宜変えてね

# 交差検証
def cross_validate(dataset, model, k=5):
    print("cross validation start.")

    all_inputs = model.padding(dataset.get_inputs()).transpose(0, 1)
    all_outputs = model.padding(dataset.get_outputs()).transpose(0, 1)

    # テンソル化してbatchsizeとseqsizeの次元を入れ替え
    # all_inputs = torch.tensor(dataset.get_inputs(), dtype=torch.float32).transpose(0, 1)
    # all_outputs = torch.tensor(dataset.get_outputs(), dtype=torch.float32).transpose(0, 1)

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
dataset = data.CoodinateData("src/prediction/data/2.mp4_pose.json")
# dataset = random_data.RandomCoodinateData()

input_size = dataset.input_dim()
output_size = dataset.output_dim()

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

#学習モデルの選択
model = CoodinatePredictionModel(input_size, output_size).to(device)

#最適化方法
optimizer = optim.Adam(model.parameters())

# 学習
for epoch in range(EPOCH_NUM):
    print("{0} / {1} epoch start.".format(epoch + 1, EPOCH_NUM))

    model.reset_state()
    optimizer.zero_grad()

    inputs = dataset.get_inputs()
    outputs = dataset.get_outputs()

    loss = model(inputs, outputs)
    loss.backward()
    optimizer.step()
    sum_loss = float(loss.data.to('cpu'))

    print("loss = {0}.".format(sum_loss))
    
    model_file = "src/prediction/trained_model/prediction_" + str(epoch + 1) + ".model"
    torch.save(model.state_dict(), model_file)

cross_validate(dataset, model, k=5)

    # srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/model_train.py 
    