# 学習モデル(lstmをベースとする)

import torch
import torch.nn as nn
from torch import cuda

# 座標の時系列データを入力として, 座標の時系列データを出力とする
class CoodinatePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super(CoodinatePredictionModel, self).__init__()
        self.W_lstm_enc = nn.LSTMCell(input_size, hidden_size)
        self.W_lstm_dec = nn.LSTMCell(output_size, hidden_size)
        self.W_hr_y = nn.Linear(hidden_size, output_size)
        
        # 出力系列長をself.output_seq_sizeとする
        self.output_seq_size = 30

        # 回帰問題なので最小二乗誤差を損失関数とする
        self.loss_fn = torch.nn.MSELoss()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.reset_state()

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def reset_state(self):
        self.hr = None

    def encode(self, inputs):
        for step in range(inputs.size(0)):
            input = inputs[step]
            self.hr = self.W_lstm_enc(input, self.hr)

    def decode(self, dec_input):
        self.hr = self.W_lstm_dec(dec_input, self.hr)
        y = self.W_hr_y(self.hr[0])
        return y

    def forward(self, inputs, outputs=None):
        inputs = inputs.to(self.device)
        if outputs is not None:
            outputs = outputs.to(self.device)
        self.encode(inputs)

        batch_size = inputs.size(1)
        self.hr = (self.hr[0], torch.zeros(batch_size, self.hidden_size).to(self.device))
        dec_input = torch.zeros(batch_size, self.output_size).to(self.device)

        if outputs is not None:
            accum_loss = 0
            for output in outputs:
                y = self.decode(dec_input)
                loss = self.loss_fn(y, output)
                accum_loss = accum_loss + loss
                dec_input = y
            return accum_loss
        else:
            result = []
            for _ in range(self.output_seq_size):
                y = self.decode(dec_input)
                result.append(y)
                dec_input = y
            result = torch.stack(result)
            return result