import json
import os
import random

train, val, test = 0.8, 0.1, 0.1
COUNT_THRESHOLD = 5

width, height = 7.32, 2.44
YOKO = 6
TATE = 2

# 定数の設定
ORIGINAL_FOLDER = "src/classification/data/original"
TRAIN_FOLDER = "src/classification/data/train"
VALID_FOLDER = "src/classification/data/valid"
TEST_FOLDER = "src/classification/data/test"

seg_width, seg_height = width / YOKO, height / TATE
filepaths = [os.path.join(ORIGINAL_FOLDER, filepath) for filepath in os.listdir(ORIGINAL_FOLDER)]
data_num = len(filepaths)
split_indices = ["train" for i in range(int(data_num * train))] + ["valid" for i in range(int(data_num * val))] + ["test" for i in range(int(data_num * test))]
random.shuffle(split_indices)
 

# まずフォルダの中身を空にする
import shutil

# フォルダの中身を空にする関数
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # フォルダ全削除
    os.makedirs(folder_path)  # フォルダを再作成

# フォルダを初期化
clear_folder(TRAIN_FOLDER)
clear_folder(VALID_FOLDER)
clear_folder(TEST_FOLDER)


for i, file_path in enumerate(filepaths):
    with open(file_path, "r") as json_open:
        frames = json.load(json_open)
    input_pose, output_pose, label = [], [], []
    label_counter = [0 for i in range(YOKO*TATE)]
    for j, frame in enumerate(frames):
        input = []
        if frame['data_type'] == 'input':
            if frame['keeper-pose'] != None:
                poses = list(frame['keeper-pose'].items())[:26]
                for pose in poses:
                    part, coords = pose
                    input.extend(coords)
                input_pose.append(input)
        elif frame['data_type'] == 'output':
            output_pose.append(frame)
            if frame['keeper-pose'] != None:
                poses = list(frame['keeper-pose'].items())[:13]
                for pose in poses:
                    part, coords = pose
                    goal_part = int((coords[1] // seg_height)*YOKO + (coords[0] // seg_width))
                    if goal_part >= 0 and goal_part < YOKO*TATE:
                        label_counter[goal_part] += 1
    label = [1 if count >= max(label_counter) else 0 for count in label_counter] # ここだけ変えた
    data = {}
    data["label"] = label
    data["input"] = input_pose
    output_filename = file_path.split("/")[len(file_path.split("/"))-1].split(".")[0] + "_dataset.json"
    if split_indices[i] == "train":
        with open(os.path.join(TRAIN_FOLDER, output_filename), "w") as file:
            json.dump(data, file, indent=4)
    elif split_indices[i] == "valid":
        with open(os.path.join(VALID_FOLDER, output_filename), "w") as file:
            json.dump(data, file, indent=4)
    else:
        with open(os.path.join(TEST_FOLDER, output_filename), "w") as file:
            json.dump(data, file, indent=4)

# poetry run python src/classification/data/make_classify_dataset.py