import torch
from torch import nn

from data import CoodinateData
from train_parameter import Train_parameter

def lstm_train(model: nn.Module, dataset: CoodinateData, train_param: Train_parameter, batch_size: int) -> None:
    input = dataset.get_input().to(train_param.device)
    target = dataset.get_dec_input().to(train_param.device)

    model.train()
    
    # ミニバッチ化
    sources = dataset.batchify(input, None, batch_size)
    targets = dataset.batchify(target, None, batch_size)

    # 各ミニバッチごとに学習する
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        train_param.optimizer.zero_grad()
        outputs = model(src, tgt.shape[0])

        loss = train_param.criterion(outputs, tgt)
        loss.backward()
        train_param.optimizer.step()

def lstm_validate(model: nn.Module, dataset: CoodinateData, train_param: Train_parameter, batch_size: int) -> float:
    input = dataset.get_input().to(train_param.device)
    target = dataset.get_dec_input().to(train_param.device)

    model.eval()
    total_loss = 0.
    
    with torch.no_grad():
        # ミニバッチ化
        sources = dataset.batchify(input, None, batch_size)
        targets = dataset.batchify(target, None, batch_size)
        # 各ミニバッチごとにモデルの出力を得て、targetと比較
        for i, (src, tgt) in enumerate(zip(sources, targets)):
            outputs = model(src, tgt.shape[0])

            loss = train_param.criterion(outputs, tgt)
            total_loss += loss.item()
            
        # 損失の平均をとる
        average_loss = total_loss / len(sources)
        return average_loss