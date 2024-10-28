# 学習モデル(lstmをベースとする)

import torch
import torch.nn as nn
from torch import cuda


# 座標の時系列データを入力として, 座標の時系列データを出力とする
class CoordinatePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=300):
        super(CoordinatePredictionModel, self).__init__()
        self.W_lstm_enc = nn.LSTMCell(input_size, hidden_size) 
        self.W_lstm_dec = nn.LSTMCell(output_size, hidden_size)
        self.W_hr_y = nn.Linear(hidden_size, output_size)
        

        # 回帰問題なので最小二乗誤差を損失関数とする
        self.loss_fn = torch.nn.MSELoss()

        # 各層の次元を保存
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

    def encode(self, input):
        self.hr = self.W_lstm_enc(input, self.hr)

    def decode(self, decode_input):
        # 1フレームのデコードを行う。LSTMで隠れ層を更新し、W_hr_y(線形変換)で予測した座標を返す
        self.hr = self.W_lstm_dec(decode_input, self.hr)
        y = self.W_hr_y(self.hr[0])
        return y


    def forward(self, inputs, outputs=None): # outputsがあれば予測値との平均二乗誤差を、なければ予測値を出力
        inputs = inputs.to(self.device)
        if outputs is not None:
            outputs = outputs.to(self.device)
        
        for input in inputs:
            self.encode(input)

        # batch_size = inputs.size(1)

        # デコードの前にメモリーセルをリセット。
        self.hr = (self.hr[0].unsqueeze(0), torch.zeros(1, self.hidden_size).to(self.device)) # LSTMセルのshapeを確認する必要があった。
        decode_input = torch.zeros(1, self.output_size).to(self.device) # 最初の値を決めておく

        if outputs is not None:
            accum_loss = 0
            for output in outputs:
                # フレームごとにデコードし、結果を元データと比較
                y = self.decode(decode_input).squeeze(0)
                # print(output.shape, y.shape)
                loss = self.loss_fn(y, output)
                accum_loss = accum_loss + loss
                decode_input = y.unsqueeze(0)
            return accum_loss
        else:
            # 予測座標を出力
            result = []
            for _ in range(30):
                y = self.decode(decode_input)
                result.append(y)
                decode_input = y
            result = torch.stack(result)
            return result