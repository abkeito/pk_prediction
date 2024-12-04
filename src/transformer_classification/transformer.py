import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

class PositionalEncoding(nn.Module):
    """
    Transformer の単語の位置情報を付与する
    Positional Encoding
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, d_input: int, d_output: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        # パラメータ設定
        self.d_model = d_model
        self.embedding = nn.Linear(d_input, d_model) # 埋め込み層
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, d_output)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor) -> Tensor:
        '''
        <args>
            src: エンコーダの入力(蹴る前のデータ) サイズは[src_seq_len, batch_size, d_input]
            tgt: 蹴った後のラベル サイズは[1, batch_size, d_output]
            src_mask: 入力にかけるマスク(基本は逆三角マスク) サイズは[src_seq_len, src_seq_len]
            src_key_padding_mask: エンコーダの入力のうちパディングによって埋めたフレームをboolで示す サイズは[batch_size, src_seq_len]
        '''
        src = self.embedding(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, src_mask, src_key_padding_mask)

        # 全時系列の情報を平均 (Global Average Pooling)
        # (seq_len, batch_size, d_model) -> (batch_size, d_model)
        memory = memory.mean(dim=0)
        output = self.decoder(memory)
        output = self.sigmoid(output)
        # 次元追加して返す
        # (batch_size, label_size) -> (1, batch_size, label_size)
        return output.unsqueeze(0)

