# 座標データのクラス

import json
import torch

class CoodinateData:
    def __init__(self, filename):
        with open(filename, "r") as json_open:
            json_load = json.load(json_open)
            self.input_list = []
            self.output_list = []
            self.input_seqsize = 60
            self.output_seqsize = 30
            self.batch_size = 1
            self.node_size = 17

            self.parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

            # for i in range(self.input_seqsize):
            #     batch_list = []
            #     for j in range(self.batch_size):
            #         coodinate_list = []
            #         for k in range(self.node_size):
            #             coodinate_list.extend(json_load[i]["keeper-pose"][self.parts[k]])
            #         batch_list.append(coodinate_list)
            #     self.input_list.append(batch_list)

            # for i in range(self.output_seqsize):
            #     if i + self.input_seqsize >= len(json_load):
            #         break
            #     batch_list = []
            #     for j in range(self.batch_size):
            #         coodinate_list = []
            #         for k in range(self.node_size):
            #             coodinate_list.extend(json_load[i + self.input_seqsize]["keeper-pose"][self.parts[k]])
            #         batch_list.append(coodinate_list)
            #     self.output_list.append(batch_list)


            for frame in json_load:
                data_type = frame["data_type"]
                if data_type is None:
                    continue
                keeper_pose = frame["keeper-pose"]
                if keeper_pose is None:
                    continue
                batch_list = []
                for i in range(self.batch_size):
                    coodinate_list = []
                    for j in range(self.node_size):
                        coodinate_list.extend(keeper_pose[self.parts[j]])
                    batch_list.append(coodinate_list)
                if data_type == "input":
                    self.input_list.append(batch_list)
                elif data_type == "output":
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
        mean = torch.mean(tensor, dim=0)
    if std is None:
        std = torch.std(tensor, dim=0, unbiased=False)
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