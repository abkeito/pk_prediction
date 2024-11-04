import json
import os
from base import ORIGINAL_FOLDER, TRAIN_FOLDER, TEST_FOLDER
import random

width, height = 7.32, 2.44
YOKO = 4
TATE = 2
COUNT_THRESHOLD = 10
seg_width, seg_height = width / YOKO, height / TATE
filepaths = [os.path.join(ORIGINAL_FOLDER, filepath) for filepath in os.listdir(ORIGINAL_FOLDER)]
data_num = len(filepaths)
split_indices = ["train" for i in range(int(data_num * 0.9))] + ["test" for i in range(data_num - int(data_num * 0.9))]
random.shuffle(split_indices)
 

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
                    if goal_part == 8 or goal_part == 9 or goal_part ==  10 or goal_part == 11:
                        goal_part -= 4
                    if goal_part >= 0 and goal_part < YOKO*TATE:
                        label_counter[goal_part] += 1
    label = [1 if count > COUNT_THRESHOLD else 0 for count in label_counter]
    data = {}
    data["label"] = label
    data["input"] = input_pose
    output_filename = file_path.split("/")[len(file_path.split("/"))-1].split(".")[0] + "_dataset.json"
    if split_indices[i] == "train":
        with open(os.path.join(TRAIN_FOLDER, output_filename), "w") as file:
            json.dump(data, file, indent=4)
    else:
        with open(os.path.join(TEST_FOLDER, output_filename), "w") as file:
            json.dump(data, file, indent=4)

# poetry run python src/classify_prediction/make_classify_dataset.py