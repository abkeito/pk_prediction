import torch
from torch import nn, optim
import copy
import matplotlib.pyplot as plt

from scripts.loader import ClassificationData
from models.transformer import TransformerModel
from models.LSTM import LSTMModel
from scripts.train import train
from scripts.validate import validate
from scripts.parameter import Train_parameter


train_dataset = ClassificationData("src/classification/data/train")
valid_dataset = ClassificationData("src/classification/data/test")

# 学習の設定
input_size = train_dataset.get_input_dim()
output_size = train_dataset.get_output_dim()
print(input_size, output_size)

# ハイパーパラメータ
d_model = 64
d_hid = 200
nlayers = 6
nhead = 2
dropout = 0.2
lr = 0.0001 # 学習率
BATCH_SIZE = 3
epoch_num = 30

train_param = Train_parameter()

# モデルの定義
models = [
    ("LSTM", LSTMModel(input_size, output_size).to(train_param.device)),
    ("Transformer", TransformerModel(d_model, input_size, output_size, nhead, d_hid, nlayers, dropout).to(train_param.device))
]
history = {}
for model_name, model in models:

    train_param.criterion = nn.BCELoss()
    train_param.optimizer = optim.Adam(model.parameters(), lr)

    # グラフ出力用
    history[model_name] = {"train loss": [], "val loss": [], "train accuracy": [], "val accuracy": []}


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
        history[model_name]["train loss"].append(train_loss)
        history[model_name]["val loss"].append(val_loss)
        history[model_name]["train accuracy"].append(train_accuracy)
        history[model_name]["val accuracy"].append(val_accuracy)


# 損失の遷移の描画
for model_name, model in models:
    plt.plot(history[model_name]["train loss"], label=f"{model_name} Train Loss")
    plt.plot(history[model_name]["val loss"], label=f"{model_name} Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.savefig("classification_loss_rate.png")
plt.close()

# 正答率の遷移の描画
for model_name, model in models:
    plt.plot(history[model_name]["train accuracy"], label=f"{model_name} Train Accuracy")
    plt.plot(history[model_name]["val accuracy"], label=f"{model_name} Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.savefig("classification_accuracy_rate.png")