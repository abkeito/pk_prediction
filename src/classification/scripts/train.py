import torch
from torch import nn

from .loader import ClassificationData
from .parameter import Train_parameter

def train(model: nn.Module, dataset: ClassificationData, train_param: Train_parameter, batch_size: int) -> None:

    input = dataset.get_input().to(train_param.device)
    input_mask = dataset.get_input_padding_mask().to(train_param.device)
    target = dataset.get_target().to(train_param.device)

    model.train() # モデルが学習モードになる
    
    # ミニバッチ化
    inputs, inputs_padding_mask = dataset.batchify(input, input_mask, batch_size)
    targets = dataset.batchify(target, None, batch_size)


    # 各ミニバッチごとに学習する
    sum_loss = 0
    sum_correct = 0
    for i, (input, input_padding_mask, target) in enumerate(zip(inputs, inputs_padding_mask, targets)):

        # ロスの計算
        train_param.optimizer.zero_grad()

        # モデルごとの出力を得る
        if model.__class__.__name__ == "TransformerModel":
            # 逆三角マスク生成（未来の情報を隠すため）
            input_mask = nn.Transformer.generate_square_subsequent_mask(input.shape[0]).to(train_param.device)
            outputs = model(input, input_mask, input_padding_mask)

        elif model.__class__.__name__ == "LSTMModel":
            outputs= model(input)     

        loss = train_param.criterion(outputs, target)
        sum_loss += loss.item()
        loss.backward()
        train_param.optimizer.step()

        # 正答率を計算
        predicted = (outputs > 0.5).float()
        sum_correct += (predicted == target).sum().item()

    # ミニバッチごとの平均損失と正答率を返す
    return sum_loss / len(inputs), sum_correct / (len(inputs) * dataset.get_output_dim() * batch_size) 