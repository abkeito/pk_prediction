import torch
from torch import nn

from .loader import ClassificationData
from .parameter import Train_parameter
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_metrics(outputs, target):
    # 二値化 (閾値 0.5)
    predicted = (outputs > 0.5).float()

    # 平均損失と正答率の計算
    sum_loss = loss_fn(outputs, target).item()
    sum_correct = (predicted == target).sum().item()
    accuracy = sum_correct / (len(target) * target.shape[1])

    # Precision, Recall, macro-F1 の計算
    # flatten して sklearn を使用 (1次元に変換)
    predicted_flat = predicted.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()
    
    precision = precision_score(target_flat, predicted_flat, average='macro', zero_division=0)
    recall = recall_score(target_flat, predicted_flat, average='macro', zero_division=0)
    macro_f1 = f1_score(target_flat, predicted_flat, average='macro', zero_division=0)

    return sum_loss / len(target), accuracy, precision, recall, macro_f1


def validate(model: nn.Module, dataset: ClassificationData, train_param: Train_parameter, batch_size: int) -> None:
    input = dataset.get_input().to(train_param.device)
    input_mask = dataset.get_input_padding_mask().to(train_param.device)
    target = dataset.get_target().to(train_param.device)

    model.eval() # モデルが評価モードになる

    sum_loss = 0.
    sum_correct = 0
    all_predicted = []
    all_targets = []

    with torch.no_grad():
        # ミニバッチ化
        inputs, inputs_padding_mask = dataset.batchify(input, input_mask, batch_size)
        targets = dataset.batchify(target, None, batch_size)

        # 各ミニバッチごとにモデルの出力を得て、targetと比較
        for i, (input, input_padding_mask, target) in enumerate(zip(inputs, inputs_padding_mask, targets)):
            # モデルごとの出力を得る
            if model.__class__.__name__ == "TransformerModel":
                # 逆三角マスク生成（未来の情報を隠すため）
                input_mask = nn.Transformer.generate_square_subsequent_mask(input.shape[0]).to(train_param.device)
                outputs = model(input, input_mask, input_padding_mask)

            elif model.__class__.__name__ == "LSTMModel":
                outputs= model(input)

            loss = train_param.criterion(outputs, target)
            sum_loss += loss.item()
            
            # 多ラベル分類問題
            # 予測の2値化
            predicted = (outputs > 0.5).float()

            # 正答率を計算
            sum_correct += (predicted == target).sum().item()

            # スコア計算ように保存
            all_predicted.append(predicted.cpu())
            all_targets.append(target.cpu())

    # Precision, Recall, macro-F1 の計算
    all_predicted = torch.cat(all_predicted, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    all_predicted = all_predicted.reshape(-1, all_predicted.shape[-1])
    all_targets = all_targets.reshape(-1, all_targets.shape[-1])

    # デバッグ用ラベルごとの結果
    # labels = range(dataset.get_output_dim())
    # precision_per_label = precision_score(all_targets, all_predicted, average=None, labels=labels, zero_division=0)
    # recall_per_label = recall_score(all_targets, all_predicted, average=None, labels=labels, zero_division=0)
    # f1_per_label = f1_score(all_targets, all_predicted, average=None, labels=labels, zero_division=0)
    # for label, precision, recall, f1 in zip(labels, precision_per_label, recall_per_label, f1_per_label):
    #     print(f"label {label}: precision {precision:.4f}, recall {recall:.4f}, f1 {f1:.4f}")

    # Precision, Recall, macro-F1 の計算
    precision = precision_score(all_targets, all_predicted, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_predicted, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_predicted, average='macro', zero_division=0)
    accuracy = sum_correct / (2*len(inputs) * dataset.get_output_dim() * batch_size - sum_correct)

    # ミニバッチごとの平均損失と正答率、Precision, Recall, macroF1を返す
    return sum_loss / len(inputs), accuracy, precision, recall, macro_f1

    #     # 多クラス分類問題
    #     # 正答率を計算
    #     predicted_indices = torch.argmax(outputs, dim=2)
    #     target_indices = torch.argmax(target, dim=2)
    #     sum_correct += (predicted_indices == target_indices).sum().item()

    # # ミニバッチごとの平均損失と正答率を返す
    # return sum_loss / len(inputs), sum_correct / (len(inputs) * batch_size) 