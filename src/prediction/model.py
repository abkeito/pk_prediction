# 学習モデル(lstmをベースとする)

import torch
import torch.nn as nn
from torch import cuda

# 座標の時系列データを入力として, 座標の時系列データを出力とする
class CodinatePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super(CodinatePredictionModel, self).__init__()
        self.W_x_hi = nn.Embedding(input_size, hidden_size)
        self.W_y_hi = nn.Embedding(output_size, hidden_size)
        self.W_lstm_enc = nn.LSTMCell(hidden_size, hidden_size)
        self.W_lstm_dec = nn.LSTMCell(hidden_size, hidden_size)
        self.W_hr_y = nn.Linear(hidden_size, output_size)
        
        # 最大出力系列長をself.max_output_sizeとする
        self.max_output_size = 60

        # 回帰問題なので最小二乗誤差を損失関数とする
        self.loss_fn = torch.nn.MSELoss()

        self.hidden_size = hidden_size
        self.reset_state()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def reset_state(self):
        self.hr = None

    def encode(self, input):
        hi = self.W_x_hi(input)
        self.hr = self.W_lstm_enc(hi, self.hr)

    def decode(self, hi):
        self.hr = self.W_lstm_dec(hi, self.hr)
        y = self.W_hr_y(self.hr[0])
        return y

    def forward(self, inputs, outputs=None):
        inputs = inputs.to(self.device)
        if outputs is not None:
            outputs = outputs.to(self.device)
        for input in inputs:
            self.encode(input)

        self.hr = (self.hr[0], torch.zeros(1, self.hidden_size).to(self.device))
        zero_token = torch.tensor(0).view(1, 1).to(self.device)
        hi = self.W_x_hi(zero_token.squeeze(0))

        if outputs is not None:
            accum_loss = 0
            for output in outputs:
                y = self.decode(hi)
                loss = self.loss_fn(y, output)
                accum_loss = accum_loss + loss
                hi = self.W_y_hi(output)
            return accum_loss
        else:
            result = []
            for _ in range(self.max_output_size):
                y = self.decode(hi)
                result.append(y)
                hi =self.W_y_hi(y)
            return result