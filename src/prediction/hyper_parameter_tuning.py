import sys
from skopt import gp_minimize
from skopt.space import Real, Integer
import torch
import torch.nn as nn
import numpy as np
import random

import data
# import random_data
from model import CoodinatePredictionModel
from pose_prediction import pose_prediction

EPOCH_NUM = 50
EPOCH_NUM_VAL = 20

# データセットの選択
dataset = data.CoodinateData('src/prediction/data/input/train')

input_size = dataset.input_dim()
output_size = dataset.output_dim()

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def cross_validate(inputs, outputs, model, optimizer, k=5):
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
        model.reset_state()
        with torch.no_grad():  # 勾配計算を無効にする
            model.eval()
            val_loss = model(val_inputs, val_outputs)
        sum_loss += val_loss.item()
    return sum_loss/k


# 目的関数の定義
def objective(params):
    hidden_size = params[0]
    learning_rate = params[1]
    
    model = CoodinatePredictionModel(input_size, output_size, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 訓練データの取得
    inputs = dataset.get_inputs()
    outputs = dataset.get_outputs()
    
    model.train()
    for epoch in range(EPOCH_NUM):  # エポック数は調整可能
        model.reset_state()
        optimizer.zero_grad()
        loss = model(inputs, outputs)
        loss.backward()
        optimizer.step()
    
    # 交差検証による評価スコアを返す
    return cross_validate(inputs, outputs, model, optimizer)

# ハイパーパラメータの探索空間
space = [
    Integer(50, 1000, name='hidden_size'),  # 隠れ層のユニット数
    Real(1e-5, 1e-1, name='learning_rate', prior='log-uniform')  # 学習率
]

# 最適化の実行
res = gp_minimize(objective, space, n_calls=50, random_state=0)

# 最適なハイパーパラメータを表示
print("Best hyperparameters:")
print(f"Hidden size: {res.x[0]}")
print(f"Learning rate: {res.x[1]}")
print(f"Best validation loss: {res.fun}")

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/prediction/hyper_parameter_tuning.py