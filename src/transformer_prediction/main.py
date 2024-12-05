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
from lstm import LSTMModel
from lstm_train import lstm_train, lstm_validate
from hyper_parameter_tuning import transformer_hyper_parameter_tuning, lstm_hyper_parameter_tuning

BATCH_SIZE = 3

epoch_num = 200

train_dataset = CoodinateData("src/transformer_prediction/data/train")
val_dataset = CoodinateData("src/transformer_prediction/data/val")
test_dataset = CoodinateData("src/transformer_prediction/data/test")

# ハイパーパラメータチューニング
transformer_hyper_params = transformer_hyper_parameter_tuning(train_dataset, val_dataset, BATCH_SIZE)
print("Best hyperparameters of transformer:")
print(f"Hidden size: {transformer_hyper_params.x[0]}")
print(f"Layer size: {transformer_hyper_params.x[1]}")
print(f"Head size: {transformer_hyper_params.x[2]}")
print(f"Dropout rate: {transformer_hyper_params.x[3]}")
print(f"Learning rate: {transformer_hyper_params.x[4]}")
print(f"Best validation loss: {transformer_hyper_params.fun}")
print("")

lstm_hyper_params = lstm_hyper_parameter_tuning(train_dataset, val_dataset, BATCH_SIZE)
print("Best hyperparameters of lstm:")
print(f"Hidden size: {lstm_hyper_params.x[0]}")
print(f"Dropout rate: {lstm_hyper_params.x[1]}")
print(f"Learning rate: {lstm_hyper_params.x[2]}")
print(f"Best validation loss: {lstm_hyper_params.fun}")
print("")

# 学習の設定
input_size = train_dataset.get_input_dim()
output_size = train_dataset.get_output_dim()

# ハイパーパラメータ
# d_hid = 200
# nlayers = 6
# nhead = 2
# dropout = 0.2
# lr = 0.0001 # 学習率

transformer_d_hid, transformer_nlayers, transformer_nhead, transformer_dropout, transformer_lr = transformer_hyper_params.x
lstm_d_hid, lstm_dropout, lstm_lr = lstm_hyper_params.x

transformer_train_param = Train_parameter()
transformer_model = TransformerModel(input_size, output_size, transformer_nhead, transformer_d_hid, transformer_nlayers, transformer_dropout).to(transformer_train_param.device)
transformer_train_param.criterion = nn.MSELoss()
transformer_train_param.optimizer = optim.Adam(transformer_model.parameters(), transformer_lr)

lstm_train_param = Train_parameter()
lstm_model = LSTMModel(input_size, output_size, lstm_d_hid, lstm_dropout).to(lstm_train_param.device)
lstm_train_param.criterion = nn.MSELoss()
lstm_train_param.optimizer = optim.Adam(lstm_model.parameters(), lstm_lr)

# グラフ出力用
history = {"transformer": [], "lstm": []}

# 学習を回す
print(f"Transformer training started.")
transformer_best_val_loss = float('inf')
transformer_best_model = None

for epoch in range(epoch_num):
    train(transformer_model, train_dataset, transformer_train_param, BATCH_SIZE)
    val_loss = validate(transformer_model, val_dataset, transformer_train_param, BATCH_SIZE)
    print(f"epoch {epoch + 1:3d} / {epoch_num} finished | val loss: {val_loss:6.4f}")

    if val_loss < transformer_best_val_loss:
        transformer_best_val_loss = val_loss
        transformer_best_model = copy.deepcopy(transformer_model)
    history["transformer"].append(val_loss)
print(f"transformer: best loss = {transformer_best_val_loss}")

print(f"LSTM training started.")
lstm_best_val_loss = float('inf')
lstm_best_model = None

for epoch in range(epoch_num):
    lstm_train(lstm_model, train_dataset, lstm_train_param, BATCH_SIZE)
    val_loss = lstm_validate(lstm_model, val_dataset, lstm_train_param, BATCH_SIZE)
    print(f"epoch {epoch + 1:3d} / {epoch_num} finished | val loss: {val_loss:6.4f}")

    if val_loss < lstm_best_val_loss:
        lstm_best_val_loss = val_loss
        lstm_best_model = copy.deepcopy(transformer_model)
    history["lstm"].append(val_loss)
print(f"lstm: best loss = {lstm_best_val_loss}")

# 損失の遷移の描画
plt.plot(history["transformer"], label="Transformer")
plt.plot(history["lstm"], label="LSTM")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validate Loss")
plt.legend()
plt.savefig("validate_loss_rate.png")

# テストデータで予測
test_enable = False
if test_enable:
    predict_and_record(transformer_best_model, test_dataset, transformer_train_param)