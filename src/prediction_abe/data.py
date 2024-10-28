# 座標データのクラス

import json
import torch
import numpy as np

class CoordinateData:
    def __init__(self, filenames):
        self.input_list = []
        self.output_list = []
        self.input_seqsize = 30
        self.output_seqsize = 30
        self.node_size = 34
        self.parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                      'nose-move', 'left_eye-move', 'right_eye-move', 'left_ear-move', 'right_ear-move', 'left_shoulder-move', 'right_shoulder-move', 'left_elbow-move', 'right_elbow-move', 'left_wrist-move', 'right_wrist-move', 'left_hip-move', 'right_hip-move', 'left_knee-move', 'right_knee-move', 'left_ankle-move', 'right_ankle-move']
        json_load = []
        for filename in filenames:
            try:
                with open(filename, "r") as json_open:
                    json_load.append(json.load(json_open))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
            except FileNotFoundError as e:
                print(f"File not found: {filename}")
        self.batch_size = len(json_load)
        
        for i in range(min(self.batch_size, len(json_load))):
            batch_list = []
            for j in range(len(json_load[i])):
                Coordinate_list = []
                if json_load[i][j]["data_type"] == "input":
                    for k in range(self.node_size):
                        if json_load[i][j]["keeper-pose"] == None:
                            Coordinate_list.extend([0,0])
                        else:
                            Coordinate_list.extend(json_load[i][j]["keeper-pose"][self.parts[k]])
                    batch_list.append(Coordinate_list)
            batch_list =  [np.tile([0, 0], self.node_size) for _ in range(self.input_seqsize - len(batch_list))] + batch_list
            self.input_list.append(batch_list)

        for i in range(min(self.batch_size, len(json_load))):
            batch_list = []
            for j in range(len(json_load[i])):
                Coordinate_list = []
                if json_load[i][j]["data_type"] == "output":
                    for k in range(self.node_size):
                        if json_load[i][j]["keeper-pose"] == None:
                            Coordinate_list.extend([0,0])
                        else:
                            Coordinate_list.extend(json_load[i][j]["keeper-pose"][self.parts[k]])
                    batch_list.append(Coordinate_list)
            batch_list = batch_list + [batch_list[len(batch_list)-1] for _ in range(self.input_seqsize - len(batch_list))]
            self.output_list.append(batch_list)

    def batch_size(self):
        return self.batch_size
    
    def input_dim(self):
        return len(self.input_list[0][0])

    def output_dim(self):
        return len(self.output_list[0][0])

    def get_inputs(self):
        return self.input_list

    def get_outputs(self):
        return self.output_list

# 標準化
def standardize(tensor, mean=None, std=None):
    if mean is None:
        mean = torch.mean(tensor)
    if std is None:
        std = torch.std(tensor, unbiased=False)
    # 0除算対策
    eps = 10**-9
    eps_tensor = torch.full_like(std, 10**-9)
    if (std < eps).all():
        std = eps_tensor
    
    standardized_tensor = (tensor - mean) / std

    # 標準化後のテンソルと、平均、標準偏差を返す
    return standardized_tensor, mean, std

# 逆標準化
def destandardize(tensor, mean, std):
    return tensor * std + mean