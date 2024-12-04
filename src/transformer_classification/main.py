import torch
from torch import nn, optim
import copy
import matplotlib.pyplot as plt

from data import ClassificationData
from transformer import TransformerModel
from train import train
from validate import validate
from train_parameter import Train_parameter
from data import ClassificationData


train_dataset = ClassificationData("src/transformer_classification/data/train")
valid_dataset = ClassificationData("src/transformer_classification/data/test")

# 学習の設定
input_size = train_dataset.get_input_dim()
output_size = train_dataset.get_output_dim()

# ハイパーパラメータ
d_model = 64
d_hid = 200
nlayers = 6
nhead = 2
dropout = 0.2
lr = 0.0001 # 学習率
BATCH_SIZE = 3
epoch_num = 200

train_param = Train_parameter()
model = TransformerModel(d_model, input_size, output_size, nhead, d_hid, nlayers, dropout).to(train_param.device)
train_param.criterion = nn.BCELoss()
train_param.optimizer = optim.Adam(model.parameters(), lr)

# グラフ出力用
history = {"train loss": [], "val loss": [], "train accuracy": [], "val accuracy": []}


# 学習を回す
best_val_loss = float('inf')
best_model = None

for epoch in range(epoch_num):
    train_loss, train_accuracy = train(model, train_dataset, train_param, BATCH_SIZE)
    val_loss, val_accuracy = validate(model, valid_dataset, train_param, BATCH_SIZE)
    print(f"epoch {epoch:3d} finished | train loss: {train_loss:6.4f}")
    print(f"epoch {epoch:3d} finished | val loss: {val_loss:6.4f}")
    print(f"epoch {epoch:3d} finished | train accuracy: {train_accuracy:6.4f}")
    print(f"epoch {epoch:3d} finished | val accuracy: {val_accuracy:6.4f}")
    history["train loss"].append(train_loss)
    history["val loss"].append(val_loss)
    history["train accuracy"].append(train_accuracy)
    history["val accuracy"].append(val_accuracy)

# 損失の遷移の描画
plt.plot(history["train loss"], label="Train Loss")
plt.plot(history["val loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.savefig("classification_loss_rate.png")
plt.close()

# 正答率の遷移の描画
plt.plot(history["train accuracy"], label="Train Accuracy")
plt.plot(history["val accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.savefig("classification_accuracy.png")
plt.close()