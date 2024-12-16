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
from hyper_parameter_tuning import HyperParameterTuning


train_dataset = ClassificationData("src/classification/data/train")
valid_dataset = ClassificationData("src/classification/data/valid")

# 学習の設定
input_size = train_dataset.get_input_dim()
output_size = train_dataset.get_output_dim()
BATCH_SIZE = 12
epoch_num = 200

train_param = Train_parameter()
tuning = HyperParameterTuning(train_dataset, valid_dataset, BATCH_SIZE)
params = {}
params["Transformer"] = tuning.transformer_hyper_parameter_tuning()
params["LSTM"] = tuning.lstm_hyper_parameter_tuning()

print("Best hyperparameters of transformer:")
print("nlayers, d_hid, nhead, dropout, lr")
print(params["Transformer"].nlayers, params["Transformer"].d_hid, params["Transformer"].nhead, params["Transformer"].dropout, params["Transformer"].lr)
print("Best hyperparameters of lstm:")
print("nlayers, d_hid, dropout, lr")
print(params["LSTM"].nlayers, params["LSTM"].d_hid, params["LSTM"].dropout, params["LSTM"].lr)

# モデルの定義
models = [
    ("LSTM", LSTMModel(input_size, output_size, params["LSTM"].nlayers, params["LSTM"].d_hid, params["LSTM"].dropout).to(train_param.device)),
    ("Transformer", TransformerModel(input_size, output_size, params["Transformer"].nhead, params["Transformer"].d_hid, params["Transformer"].nlayers, params["Transformer"].dropout).to(train_param.device))
]
history = {}
best_case = {}
for model_name, model in models:

    train_param.criterion = nn.BCELoss()
    train_param.optimizer = optim.Adam(model.parameters(), params[model_name].lr)

    # グラフ出力用
    history[model_name] = {"train loss": [], "val loss": [], "train accuracy": [], "val accuracy": []}


    # 学習を回す
    best_model_state = None
    best_train_loss = float('inf')  # 初期値を無限大に設定
    best_epoch = -1  # 初期値を -1 に設定

    for epoch in range(epoch_num):
        train_loss, train_accuracy = train(model, train_dataset, train_param, BATCH_SIZE)
        val_loss, val_accuracy = validate(model, valid_dataset, train_param, BATCH_SIZE)
        if epoch % 5 == 0:
            print(f"epoch {epoch:3d} finished | train loss: {train_loss:6.4f}")
            print(f"epoch {epoch:3d} finished | val loss: {val_loss:6.4f}")
            print(f"epoch {epoch:3d} finished | train accuracy: {train_accuracy:6.4f}")
            print(f"epoch {epoch:3d} finished | val accuracy: {val_accuracy:6.4f}")
        history[model_name]["train loss"].append(train_loss)
        history[model_name]["val loss"].append(val_loss)
        history[model_name]["train accuracy"].append(train_accuracy)
        history[model_name]["val accuracy"].append(val_accuracy)

        # 現在のモデルがベストの train_loss なら保存
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_model_state = model.state_dict()  # モデルの状態を保存
            best_epoch = epoch


    # ベストモデルのテスト
    model.load_state_dict(best_model_state)
    final_val_loss, final_val_accuracy = validate(model, valid_dataset, train_param, BATCH_SIZE)
    best_case[model_name] = {"epoch": best_epoch, "val loss": final_val_loss, "val accuracy": final_val_accuracy}

# ベストモデルの出力
for model_name, case in best_case.items():
    print(f"Best model of {model_name}:")
    print(f"epoch: {case['epoch']}, val loss: {case['val loss']}, val accuracy: {case['val accuracy']}")

# 損失の遷移の描画
for model_name, model in models:
    plt.plot(history[model_name]["train loss"], label=f"{model_name} Train Loss")
    plt.plot(history[model_name]["val loss"], label=f"{model_name} Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss")
plt.legend()
plt.savefig("classification_loss.png")
plt.close()

# 正答率の遷移の描画
for model_name, model in models:
    plt.plot(history[model_name]["train accuracy"], label=f"{model_name} Train Accuracy")
    plt.plot(history[model_name]["val accuracy"], label=f"{model_name} Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy")
plt.legend()
plt.savefig("classification_accuracy.png")