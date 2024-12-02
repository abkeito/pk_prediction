import torch
from torch import nn
import json

from data import CoodinateData
from train_parameter import Train_parameter

def test(model: nn.Module, dataset: CoodinateData, train_param: Train_parameter) -> tuple:
    input = dataset.get_input().to(train_param.device)
    decoder_input = dataset.get_dec_input().to(train_param.device)
    target = dataset.get_target().to(train_param.device)
    input_mask = dataset.get_input_padding_mask().to(train_param.device)
    decoder_input_mask = dataset.get_dec_input_padding_mask().to(train_param.device)

    model.eval()
    
    with torch.no_grad():
        # 逆三角マスク生成
        src_mask = nn.Transformer.generate_square_subsequent_mask(input.shape[0]).to(train_param.device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(decoder_input.shape[0]).to(train_param.device)

        output = model(input, decoder_input, src_mask, tgt_mask, input_mask, decoder_input_mask)

        loss = train_param.criterion(output, target)
    
    return output, loss

        


def predict_and_record(model: nn.Module, dataset: CoodinateData, train_param: Train_parameter) -> None:
    output, test_loss = test(model, dataset, train_param)
    print(f"Test Loss: {test_loss}")

    keypoints = dataset.parts
    keypoints_size = dataset.node_size

    # 入出力を[batch_size, seq_len, d_model]の形状にする
    input = dataset.get_input().permute(1, 0, 2)
    output = output.permute(1, 0, 2)

    for i, (input_seq, output_seq) in enumerate(zip(input, output)):
        frame_id = 0
        predicted_coodinates = []
        for frame in input_seq:
            coodinates = torch.reshape(frame, (keypoints_size, 2))
            keeper_pose = dict(zip(keypoints, coodinates.tolist()))

            keeper_pose = {
                "frame_id" : frame_id,
                "data_type" : "input",
                "keeper-pose" : keeper_pose
            }

            predicted_coodinates.append(keeper_pose)
            frame_id += 1

        for frame in output_seq:
            coodinates = torch.reshape(frame, (keypoints_size, 2))
            keeper_pose = dict(zip(keypoints, coodinates.tolist()))

            frame = {
                "frame_id" : frame_id,
                "data_type" : "output",
                "keeper-pose" : keeper_pose
            }

            predicted_coodinates.append(frame)
            frame_id += 1

        # 予測結果をjsonファイルに記録
        input_files = dataset.input_files
        # 各ファイル名は"{num}_dataset.json"の形式である必要がある。出力ファイルは"{num}_predicrion.json"とする
        with open("src/transformer_prediction/predicted/" + input_files[i][:-12] + "prediction.json", "w") as f:
            json.dump(predicted_coodinates, f, indent=4)

        print(f"Predicted action of the keeper saved to src/transformer_prediction/predicted/" + input_files[i][:-12] + "prediction.json")
    