import torch
from torch import nn, Tensor

# 座標の時系列データを入力として, 座標の時系列データを出力とする
class LSTMModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, d_hid: int, dropout: float) -> None:
        super().__init__()
        self.model_type = 'LSTM'
        self.input_size = input_size
        self.output_size = output_size
        self.d_hid = d_hid
        self.dropout = dropout

        self.lstm_encoder = nn.LSTMCell(self.input_size, self.d_hid)
        self.lstm_decoder = nn.LSTMCell(self.output_size, self.d_hid)
        self.Dropout = nn.Dropout(self.dropout) # 過学習を防ぐため、一定割合でユニットを削除
        self.decoder = nn.Linear(self.d_hid, self.output_size)

        self.reset_state()

    # 隠れ層をリセット
    def reset_state(self) -> None:
        self.hr = None

    def encode(self, inputs: Tensor) -> None:
        # フレームごとにLSTMで隠れ層の値を更新
        for input in inputs:
            self.hr = self.lstm_encoder(input, self.hr)

    def decode(self, dec_inputs: Tensor) -> Tensor:
        # 1フレームのデコードを行う。LSTMで隠れ層を更新し、W_hr_y(線形変換)で予測した座標を返す
        self.hr = self.lstm_decoder(dec_inputs, self.hr)
        y = self.decoder(self.hr[0])
        return y
        
    def forward(self, inputs: Tensor, dec_seq_len: int) -> Tensor:
        # エンコード
        self.encode(inputs)

        # デコードの前にセル状態をリセット
        self.reset_state()

        dec_input = inputs[-1]
        # デコードにより予測座標を出力
        result = []
        for _ in range(dec_seq_len):
            y = self.decode(dec_input)
            y = self.Dropout(y)
            result.append(y)
            dec_input = y
        result = torch.stack(result)
        return result