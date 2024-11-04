# 座標データのクラス

import os
import json
import torch

<<<<<<< HEAD
class CoordinateData:
    def __init__(self, filenames):
        self.input_list = []
        self.output_list = []
        self.input_seqsize = 15
        self.output_seqsize = 15
        self.batch_size = 5
        self.node_size = 17
        self.parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

        json_load = []
        for filename in filenames:
            try:
                with open(filename, "r") as json_open:
                    json_load.append(json.load(json_open))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {filename}: {e}")
            except FileNotFoundError as e:
                print(f"File not found: {filename}")

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
            self.output_list.append(batch_list)
=======
class CoodinateData:
    def __init__(self, dirname):
        self.input_list = []
        self.output_list = []

        self.parts = ['face', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'face-move', 'left_shoulder-move', 'right_shoulder-move', 'left_elbow-move', 'right_elbow-move', 'left_wrist-move', 'right_wrist-move', 'left_hip-move', 'right_hip-move', 'left_knee-move', 'right_knee-move', 'left_ankle-move', 'right_ankle-move']
        self.node_size = len(self.parts)

        self.input_dir = dirname
        self.input_files = [f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]

        self.batch_size = len(self.input_files)

        # 各データをまとめてバッチ化
        for i in range(self.batch_size):
            # jsonファイルを順に読み込み
            # 数字を変えれば使うデータを変えられる
            with open(os.path.join(self.input_dir, self.input_files[i]), "r") as json_open:
                json_load = json.load(json_open)
                input_frame_list = []
                output_frame_list = []
                # フレームごとに座標の情報をリストに格納
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
                # 座標情報をテンソルにしてからリストに追加
                input_frame_tensor = torch.tensor(input_frame_list, dtype=torch.float32)
                output_frame_tensor = torch.tensor(output_frame_list, dtype=torch.float32)
                # テンソルが空でないことを確認
                if input_frame_tensor.size(0) > 0 and output_frame_tensor.size(0) > 0:
                    self.input_list.append(input_frame_tensor)
                    self.output_list.append(output_frame_tensor)

>>>>>>> main

    def get_batch_size(self):
        return len(self.input_list)
    
    def input_dim(self):
        return len(self.input_list[0][0])

    def output_dim(self):
        return len(self.output_list[0][0])

    def get_inputs(self):
        return self.input_list

    def get_outputs(self):
        return self.output_list

<<<<<<< HEAD
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
=======
    def get_input_files(self):
        return self.input_files
>>>>>>> main

    # 総フレーム数
    def out_seq_len(self):
        res = 0
        for output in self.output_list:
            res += output.size(0)
        return res
