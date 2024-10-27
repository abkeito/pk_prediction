# 学習モデル(lstmをベースとする)

import torch
import torch.nn as nn
from torch import cuda

# 座標の時系列データを入力として, 座標の時系列データを出力とする
class CoordinatePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super(CoordinatePredictionModel, self).__init__()
        self.W_lstm_enc = nn.LSTMCell(input_size, hidden_size)
        self.W_lstm_dec = nn.LSTMCell(output_size, hidden_size)
        self.W_hr_y = nn.Linear(hidden_size, output_size)
        
        # 出力系列長をself.output_seq_sizeとする
        self.output_seq_size = 15

        # 回帰問題なので最小二乗誤差を損失関数とする
        self.loss_fn = torch.nn.MSELoss()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.reset_state()

        # デバイスの設定
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    # 隠れ層をリセット
    def reset_state(self):
        self.hr = None

    def encode(self, inputs):
        # フレームごとにLSTMで隠れ層の値を更新
        for frame in range(inputs.size(0)): #テンソルの .size(0) という attribute
            input = inputs[frame]
            self.hr = self.W_lstm_enc(input, self.hr)

    def decode(self, dec_input):
        # 1フレームのデコードを行う。LSTMで隠れ層を更新し、W_hr_y(線形変換)で予測した座標を返す
        self.hr = self.W_lstm_dec(dec_input, self.hr)
        y = self.W_hr_y(self.hr[0])
        return y

    def forward(self, inputs, outputs=None): # outputsがあれば予測値との平均二乗誤差を、なければ予測値を出力
        inputs = inputs.to(self.device)
        if outputs is not None:
            outputs = outputs.to(self.device)
        
        self.encode(inputs)

        # batch_size = inputs.size(1)

        # デコードの前にセル状態をリセット
        self.hr = (self.hr[0], torch.zeros(1, self.hidden_size).to(self.device).squeeze(0))
        dec_input = torch.zeros(1, self.output_size).to(self.device).squeeze(0)

        if outputs is not None:
            accum_loss = 0
            for output in outputs:
                # フレームごとにデコードし、結果を元データと比較
                y = self.decode(dec_input)
                loss = self.loss_fn(y, output)
                accum_loss += loss
                dec_input = y
            return accum_loss
        else:
            # 予測座標を出力
            result = []
            for _ in range(self.output_seq_size):
                y = self.decode(dec_input)
                result.append(y)
                dec_input = y
            result = torch.stack(result)
            return result