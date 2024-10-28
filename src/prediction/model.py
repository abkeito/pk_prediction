# 学習モデル(lstmをベースとする)

import torch
import torch.nn as nn
from torch import cuda
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_sequence, pack_padded_sequence

# 座標の時系列データを入力として, 座標の時系列データを出力とする
class CoodinatePredictionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=100):
        super(CoodinatePredictionModel, self).__init__()
        self.W_lstm_enc = nn.LSTMCell(input_size, hidden_size)
        self.W_lstm_dec = nn.LSTMCell(output_size, hidden_size)
        self.Dropout = nn.Dropout(0.2) # 過学習を防ぐため、一定割合でユニットを削除
        self.W_hr_y = nn.Linear(hidden_size, output_size)
        
        # 出力系列長をself.output_seq_sizeとする
        self.output_seq_size = 30

        # 回帰問題なので最小二乗誤差を損失関数とする
        self.loss_fn = torch.nn.MSELoss()

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

    # パディング
    def padding(self, batch_list):
        # 各バッチの系列長を配列で保存
        framelen_list = [len(batch) for batch in batch_list]
        # リストをPackedSequenceに変換し、0埋めにより系列長を揃える
        batch_list_packed = nn.utils.rnn.pack_sequence(batch_list, enforce_sorted=False)
        batch_list_padded, _ = nn.utils.rnn.pad_packed_sequence(batch_list_packed, batch_first=True, padding_value=0.)
        return batch_list_padded

    # PackedSequenceオブジェクトを元に戻す
    def unpack(self, y):
        y, _ = pad_packed_sequence(y, batch_first=True, padding_value=0., total_length=None)
        return y

    # 座標の平均と標準偏差を出力
    def get_mean_and_std(self, inputs):
        padded_inputs = self.padding(inputs)
        # 0を除外して標準化を行う
        mask = (padded_inputs != 0.).any(dim=2)
        masked_tensor = padded_inputs[mask]
        mean = torch.mean(masked_tensor, dim=0).to(self.device)
        std = torch.std(masked_tensor, dim=0, unbiased=False).to(self.device)
        return mean, std

    # 標準化
    def standardize(self, tensor, mean, std):
        # 0除算対策
        eps = 10**-9
        eps_tensor = torch.full_like(std, 10**-9)
        if (std < eps).any():
            std = eps_tensor
        
        standardized_tensor = (tensor - mean) / std

        # 標準化後のテンソルを返す
        return standardized_tensor

    # 逆標準化
    def destandardize(self, tensor, mean, std):
        return tensor * std + mean

    def encode(self, inputs):
        # フレームごとにLSTMで隠れ層の値を更新
        for frame in range(inputs.size(0)):
            input = inputs[frame]
            self.hr = self.W_lstm_enc(input, self.hr)
            self.hr = (self.Dropout(self.hr[0]), self.hr[1]) # ドロップアウト

    def decode(self, dec_input):
        # 1フレームのデコードを行う。LSTMで隠れ層を更新し、W_hr_y(線形変換)で予測した座標を返す
        self.hr = self.W_lstm_dec(dec_input, self.hr)
        self.hr = (self.Dropout(self.hr[0]), self.hr[1]) # ドロップアウト
        y = self.W_hr_y(self.hr[0])
        return y

    def forward(self, inputs, outputs=None): # outputsがあれば予測値との平均二乗誤差を、なければ予測値を出力
        mean, std = self.get_mean_and_std(inputs)
        # batchごとに標準化
        standardized_inputs = []
        for input in inputs:
            input = input.to(self.device)
            standardized_input = self.standardize(input, mean, std)
            standardized_inputs.append(standardized_input)
        # パディング
        padded_inputs = self.padding(standardized_inputs)
        inputs = padded_inputs.to(self.device)

        # (batch_size, seq_len, 34) -> (seq_len, batch_size, 34)に次元の入れ替えを行う（LSTMCellモデルに通すため）
        inputs = torch.permute(inputs, (1, 0, 2))

        if outputs is not None:
            # 標準化
            standardized_outputs = []
            for output in outputs:
                output = output.to(self.device)
                standardized_output = self.standardize(output, mean, std)
                standardized_outputs.append(standardized_output)
            # パディング
            padded_outputs = self.padding(standardized_outputs)
            outputs = padded_outputs.to(self.device)
            # 次元の入れ替え
            outputs = torch.permute(outputs, (1, 0, 2))
        
        self.encode(inputs)

        batch_size = inputs.size(1)

        # デコードの前にセル状態をリセット
        self.hr = (self.hr[0], torch.zeros(batch_size, self.hidden_size).to(self.device))
        dec_input = torch.zeros(batch_size, self.output_size).to(self.device)

        if outputs is not None:
            accum_loss = 0
            for output in outputs:
                # フレームごとにデコードし、結果を元データと比較
                y = self.decode(dec_input)
                loss = self.loss_fn(y, output)
                accum_loss = accum_loss + loss
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
            # 最後に逆標準化して出力
            result = self.destandardize(result, mean, std)
            return result