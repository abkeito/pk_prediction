import torch
from torch import nn, optim
import copy
import matplotlib.pyplot as plt

from data import CoodinateData
from transformer import TransformerModel
from train import train
from validate import validate
from test_and_record import predict_and_record
from train_parameter import Train_parameter

BATCH_SIZE = 3

epoch_num = 200

train_dataset = CoodinateData("src/transformer_prediction/data/train")
val_dataset = CoodinateData("src/transformer_prediction/data/val")
test_dataset = CoodinateData("src/transformer_prediction/data/test")

# 学習の設定
input_size = train_dataset.get_input_dim()
output_size = train_dataset.get_output_dim()

# ハイパーパラメータ
d_hid = 200
nlayers = 6
nhead = 2
dropout = 0.2
lr = 0.0001 # 学習率

train_param = Train_parameter()
model = TransformerModel(input_size, output_size, nhead, d_hid, nlayers, dropout).to(train_param.device)
train_param.criterion = nn.MSELoss()
train_param.optimizer = optim.Adam(model.parameters(), lr)

# グラフ出力用
history = {"val loss": []}

# 学習を回す
best_val_loss = float('inf')
best_model = None

for epoch in range(epoch_num):
    train(model, train_dataset, train_param, BATCH_SIZE)
    val_loss = validate(model, val_dataset, train_param, BATCH_SIZE)
    print(f"epoch {epoch:3d} finished | val loss: {val_loss:6.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
    history["val loss"].append(val_loss)

# 損失の遷移の描画
plt.plot(history["val loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validate Loss")
plt.savefig("validate_loss_rate.png")

# テストデータで予測
predict_and_record(best_model, test_dataset, train_param)