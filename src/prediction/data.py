# 座標データのクラス

import json
import torch

class CoodinateData:
    def __init__(self, filename):
        self.input_list = []
        self.output_list = []
        self.batch_size = 10
        self.node_size = 17

        self.parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

        for i in range(self.batch_size):
            with open(filename, "r") as json_open:
                json_load = json.load(json_open)
                input_frame_list = []
                output_frame_list = []
                for frame in json_load:
                    data_type = frame["data_type"]
                    if data_type is None:
                        continue
                    keeper_pose = frame["keeper-pose"]
                    if keeper_pose is None:
                        continue
                    coodinate_list = []
                    for j in range(self.node_size):
                        # 読み取れなかったノードはゴールの中心にあると仮定
                        if keeper_pose[self.parts[j]] == [0., 0.]:
                            coodinate_list.extend([3.66, 1.22]) # ゴールの中心
                        else:
                            coodinate_list.extend(keeper_pose[self.parts[j]])
                    if data_type == "input":
                        input_frame_list.append(coodinate_list)
                    elif data_type == "output":
                        output_frame_list.append(coodinate_list)
                input_frame_tensor = torch.tensor(input_frame_list, dtype=torch.float32)
                output_frame_tensor = torch.tensor(output_frame_list, dtype=torch.float32)
                self.input_list.append(input_frame_tensor)
                self.output_list.append(output_frame_tensor)


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
