import torch
from torch import nn

from data import CoodinateData
from train_parameter import Train_parameter

def train(model: nn.Module, dataset: CoodinateData, train_param: Train_parameter, batch_size: int) -> None:
    input = dataset.get_input().to(train_param.device)
    decoder_input = dataset.get_dec_input().to(train_param.device)
    target = dataset.get_target().to(train_param.device)
    input_mask = dataset.get_input_padding_mask().to(train_param.device)
    decoder_input_mask = dataset.get_dec_input_padding_mask().to(train_param.device)

    model.train()
    
    # ミニバッチ化
    sources, sources_padding_mask = dataset.batchify(input, input_mask, batch_size)
    dec_inputs, targets_padding_mask = dataset.batchify(decoder_input, decoder_input_mask, batch_size)
    targets = dataset.batchify(target, None, batch_size)

    # 各ミニバッチごとに学習する
    for i, (src, dec_input, tgt, src_padding_mask, tgt_padding_mask) in enumerate(zip(sources, dec_inputs, targets, sources_padding_mask, targets_padding_mask)):
  
        # 逆三角マスク生成（未来の情報を隠すため）
        src_mask = nn.Transformer.generate_square_subsequent_mask(src.shape[0]).to(train_param.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_input.shape[0]).to(train_param.device)

        train_param.optimizer.zero_grad()
        outputs = model(src, dec_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)

        loss = train_param.criterion(outputs, tgt)
        loss.backward()
        train_param.optimizer.step()