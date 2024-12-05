from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import torch
from torch import nn, optim

from transformer import TransformerModel
from lstm import LSTMModel
from train_parameter import Train_parameter
from train import train
from validate import validate
from lstm_train import lstm_train, lstm_validate

def transformer_hyper_parameter_tuning(train_dataset, val_dataset, batch_size):
    input_size = train_dataset.get_input_dim()
    output_size = train_dataset.get_output_dim()

    def objective(params):
        d_hid, nlayers, nhead, dropout, lr = params
        train_param = Train_parameter()
        model = TransformerModel(input_size, output_size, int(nhead), int(d_hid), int(nlayers), dropout).to(train_param.device)
        train_param.criterion = nn.MSELoss()
        train_param.optimizer = optim.Adam(model.parameters(), lr)

        best_val_loss = 100.0 # 十分大きな値
        for _ in range(20):  # 短いスパンでテスト
            train(model, train_dataset, train_param, batch_size)
            val_loss = validate(model, val_dataset, train_param, batch_size)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        return best_val_loss

    param_space = [
        Integer(50, 1000, name='d_hid'),
        Integer(2, 10, name='nlayers'),
        Categorical([2, 4], name='nhead'), # nheadはinput_size = 52の約数かつ2の累乗でなくてはならない
        Real(0.1, 0.5, name='dropout'),
        Real(1e-5, 1e-1, name='learning_rate', prior='log-uniform')
    ]

    result = gp_minimize(objective, param_space, n_calls=50, random_state=0)
    return result

def lstm_hyper_parameter_tuning(train_dataset, val_dataset, batch_size):
    input_size = train_dataset.get_input_dim()
    output_size = train_dataset.get_output_dim()

    def objective(params):
        d_hid, dropout, lr = params
        train_param = Train_parameter()
        model = LSTMModel(input_size, output_size, int(d_hid), dropout).to(train_param.device)
        train_param.criterion = nn.MSELoss()
        train_param.optimizer = optim.Adam(model.parameters(), lr)

        best_val_loss = float('inf')
        for _ in range(5):  # 短いスパンでテスト
            lstm_train(model, train_dataset, train_param, batch_size)
            val_loss = lstm_validate(model, val_dataset, train_param, batch_size)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        return best_val_loss
    
    param_space = [
        Integer(50, 1000, name='d_hid'),
        Real(0.1, 0.5, name='dropout'),
        Real(1e-5, 1e-1, name='learning_rate', prior='log-uniform')
    ]

    result = gp_minimize(objective, param_space, n_calls=50, random_state=0)
    return result