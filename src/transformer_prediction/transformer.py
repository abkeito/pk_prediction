import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import math

# 入力ベクトルの位置情報
class PositionalEncoding(nn.Module):
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
        # xは[seq_len, batch_size, embedding_dim]の形状のTensor
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, d_model: int, ntoken: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        decoder_layers = TransformerDecoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    # 重み付けの初期化
    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_key_padding_mask: Tensor, tgt_key_padding_mask: Tensor) -> Tensor:
        '''
        <args>
            src: エンコーダの入力(蹴る前のデータ) サイズは[src_seq_len, batch_size, dim_model]
            tgt: デコーダの入力(蹴った後のデータ) サイズは[tgt_seq_len, batch_size, dim_model]
            src_mask: 入力にかけるマスク(基本は逆三角マスク) サイズは[src_seq_len, src_seq_len]
            tgt_mask: デコーダの入力にかけるマスク(基本は逆三角マスク) サイズは[tgt_seq_len, tgt_seq_len]
            src_key_padding_mask: エンコーダの入力のうちパディングによって埋めたフレームをboolで示す サイズは[batch_size, src_seq_len]
            tgt_key_padding_mask: デコーダの入力のうちパディングによって埋めたフレームをboolで示す サイズは[batch_size, tgt_seq_len]
        '''
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        memory = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        output = self.transformer_decoder(tgt, memory, tgt_mask, None, tgt_key_padding_mask, src_key_padding_mask)
        output = self.decoder(output)
        return output