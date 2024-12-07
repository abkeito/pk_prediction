import torch 
import torch.nn as nn

gpu_id = "cuda"

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, num_layers: int = 5, hidden_size: int = 256, dropout=0.6):
        super(LSTMModel, self).__init__()

        if gpu_id is not None:
            self.device = torch.device(gpu_id)
            self.to(self.device)
        else:
            self.device = torch.device('cpu')


        # 双方向LSTM層 + ドロップアウト
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        
        # LSTMの出力を正規化
        self.layer_norm = nn.LayerNorm(hidden_size * 2) # 双方向なので隠れ層のサイズは2倍

        # 全結合層
        # 全結合層（中間層を追加）
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # シグモイド層（多ラベル分類用）
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # LSTMの隠れ状態を初期化
        h0 = torch.zeros(2*self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)
        c0 = torch.zeros(2*self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(self.device)
        
        # LSTMの順伝播
        out, _ = self.lstm(x.to(self.device), (h0, c0))

        # LSTMの最後の出力を取得し、正規化
        out = self.layer_norm(out[-1, :, :])

        # 全結合層に通す
        out = self.fc1(out)
        out = nn.ReLU()(out)  # 中間層でReLUを適用
        out = self.fc2(out)

        # シグモイド層で多ラベルの出力
        out = self.sigmoid(out)

        # (batch_size, label_size) -> (1, batch_size, label_size)
        return out.unsqueeze(0)