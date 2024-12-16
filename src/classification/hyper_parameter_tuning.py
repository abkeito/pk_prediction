from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
import torch
from torch import nn, optim

from models.transformer import TransformerModel
from models.LSTM import LSTMModel
from scripts.parameter import Train_parameter
from scripts.train import train
from scripts.validate import validate

class HyperParameterTransformer:

    def __init__(self, param):
        self.d_hid = param.x[0]
        self.nlayers = param.x[1]
        self.nhead = param.x[2]
        self.dropout = param.x[3]
        self.lr = param.x[4]

class HyperParameterLSTM:
           
    def __init__(self, param):
        self.nlayers = param.x[0]
        self.d_hid = param.x[1]
        self.dropout = param.x[2]
        self.lr = param.x[3]

class HyperParameterTuning:
    def __init__(self, train_dataset, val_dataset, batch_size):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.input_size = train_dataset.get_input_dim()
        self.output_size = train_dataset.get_output_dim()

    def transformer_hyper_parameter_tuning(self):

        def objective(params):
            d_hid, nlayers, nhead, dropout, lr = params
            train_param = Train_parameter()
            model = TransformerModel(self.input_size, self.output_size, int(nhead), int(d_hid), int(nlayers), dropout).to(train_param.device)
            train_param.criterion = nn.BCELoss()
            train_param.optimizer = optim.Adam(model.parameters(), lr)

            best_val_loss = 100.0 # 十分大きな値
            for _ in range(50):  # 短いスパンでテスト
                train(model, self.train_dataset, train_param, self.batch_size)
                val_loss = validate(model, self.val_dataset, train_param, self.batch_size)[0]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            return best_val_loss

        param_space = [
            Integer(50, 500, name='d_hid'),
            Integer(2, 10, name='nlayers'),
            Categorical([2, 4], name='nhead'), # nheadはinput_size = 52の約数かつ2の累乗でなくてはならない
            Real(0.1, 0.5, name='dropout'),
            Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform')
        ]

        result = gp_minimize(objective, param_space, n_calls=10, random_state=0, verbose=True)

        return HyperParameterTransformer(result)

    def lstm_hyper_parameter_tuning(self):

        def objective(params):
            nlayers, d_hid, dropout, lr = params
            train_param = Train_parameter()
            model = LSTMModel(self.input_size, self.output_size, int(nlayers), int(d_hid), dropout).to(train_param.device)
            train_param.criterion = nn.BCELoss()
            train_param.optimizer = optim.Adam(model.parameters(), lr)

            best_val_loss = float('inf')
            for _ in range(50):  # 短いスパンでテスト
                train(model, self.train_dataset, train_param, self.batch_size)
                val_loss = validate(model, self.val_dataset, train_param, self.batch_size)[0]
                if val_loss < best_val_loss:
                    best_val_loss = val_loss

            return best_val_loss
        
        param_space = [
            Integer(2, 10, name='nlayers'),
            Integer(50, 1000, name='d_hid'),
            Real(0.1, 0.5, name='dropout'),
            Real(1e-5, 1e-3, name='learning_rate', prior='log-uniform')
        ]

        result = gp_minimize(objective, param_space, n_calls=10, random_state=0, verbose=True)

        return HyperParameterLSTM(result)