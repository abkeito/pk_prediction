import torch.nn as nn
from torch import optim
from torch import cuda
import os, json
import wandb
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import load_data, preprocess_data, get_edge_matrix
from model import MyModel

# ハイパーパラメータ！！！
EPOCH_NUM = 500
learning_rate = 1e-6

# 0. GPU 関係
# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(torch.version.cuda)
print(torch.__version__)
device_availability = 'gpu' if torch.cuda.is_available() else 'cpu'
print(device_availability)

# wandb を設定
wandb.init(
    # プロジェクト名を設定
    project="PK-classify-project", 
    # 実行名をつける(これまでのように、指定しない場合は、自動で作成してくれます）
    # ハイパーパラメータ等の情報を保存
    config={
    "learning_rate": learning_rate,
    "architecture": "LSTM",
    "dataset": "dataset",
    "epochs": EPOCH_NUM,
    })

# 1. データセットをとってくる
folder = "/home/u01170/AI_practice/pk_prediction/src/classify_prediction/data/train"
filepaths = [os.path.join(folder, filepath) for filepath in os.listdir(folder)]
X_train, X_valid, Y_train, Y_valid = load_data(filepaths)
edge_list = torch.tensor(get_edge_matrix()).T

# 2. DataLoader に突っ込む
train_loader = preprocess_data(X_train, Y_train)
valid_loader = preprocess_data(X_valid, Y_valid)

# 3. モデルを作る
model = MyModel(input_size=52, output_size=8).to(device)
print(model)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4. 学習する
for epoch in range(EPOCH_NUM):
    print("{0} / {1} epoch start.".format(epoch + 1, EPOCH_NUM))
    sum_loss, correct = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data, edge_list)
        loss = criterion(output, target.to(device))
        sum_loss += loss
        # 予測ラベルを作成
        predicted = (output > 0.5).float()  # 0.5をしきい値として使用
        # predicted と target はともに配列として処理されると仮定
        correct += (predicted == target.to(device).float()).sum()  # 正解の数

        loss.backward()
        optimizer.step()

    print(correct, batch_idx)

    train_accuracy = correct / ((batch_idx+1) * 8) # 配列の全要素に対する正解率
    sum_loss = sum_loss / (batch_idx+1)
    wandb.log({"train_loss": sum_loss})
    wandb.log({"train_accuracy": train_accuracy})
    print(f"Epoch [{epoch+1}/{EPOCH_NUM}], Loss: {sum_loss}")
    print(f"Epoch [{epoch+1}/{EPOCH_NUM}], Accuracy: {train_accuracy}")

    if (epoch+1) % 10 == 0:
        valid_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(valid_loader):
            output = model(data, edge_list)
            valid_loss += criterion(output, target.to(device))
            # 予測ラベルを作成
            predicted = (output > 0.5).float()  # 0.5をしきい値として使用
            # predicted と target はともに配列として処理されると仮定
            correct += (predicted == target.to(device).float()).sum()  # 正解の数

        valid_accuracy = correct / ((batch_idx+1) * 8) # 配列の全要素に対する正解率
        print(f"Epoch [{epoch+1}/{EPOCH_NUM}], Valid Loss: {valid_loss.item() / (batch_idx+1)}")
        print(f"Epoch [{epoch+1}/{EPOCH_NUM}], Accuracy: {valid_accuracy}")
        wandb.log({"valid_loss": valid_loss.item() / (batch_idx+1)})
        wandb.log({"valid_accuracy": valid_accuracy})
        model_file = "src/classify_prediction/trained_model/prediction_" + str(epoch + 1) + ".model"
        torch.save(model.state_dict(), model_file)

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/classify_prediction/model_train.py