# test
import sys
import torch
import os, json

from data import load_data, preprocess_data, get_edge_matrix
from model import MyModel

EPOCH_NUM = 300

TEST_FOLDER_PATH = "/home/u01170/AI_practice/pk_prediction/src/classify_prediction/data/test"
RESULT_FOLDER_PATH = "/home/u01170/AI_practice/pk_prediction/src/classify_prediction/test_result"
filenames = os.listdir(TEST_FOLDER_PATH)

for filename in filenames:
    # データセットの選択
    filepath = os.path.join(TEST_FOLDER_PATH, filename)

    with open(filepath, "r") as f:
        data = json.load(f)
    X = [data["input"]]
    Y = [data["label"]]
    test_loader = preprocess_data(X, Y)

    # デバイスの設定
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # モデルの選択
    model = MyModel(52, 8).to(device)
    edge_list = torch.tensor(get_edge_matrix()).T

    model.load_state_dict(torch.load("src/classify_prediction/trained_model/prediction_{0}.model".format(EPOCH_NUM)))

    for batch_idx, (data, target) in enumerate(test_loader):
        output = {"result":model(data, edge_list).tolist()}
    output["label"] = Y
    # これをsave
    result_path = os.path.join(RESULT_FOLDER_PATH, filename.split("_")[0] + "_result.json")
    with open(result_path, 'w') as f:
        json.dump(output, f, indent=4)

    print(f"Data has been saved to {result_path}")
    

    # 出力をファイルに保存
    # pose_prediction(inputs[0], outputs, os.path.join("/home/u01170/AI_practice/pk_prediction/src/prediction_abe/prediction_data", str(index) + "predict_pose.json"))

# srun -p p -t 10:00 --gres=gpu:1 --pty poetry run python src/classify_prediction/model_test.py
