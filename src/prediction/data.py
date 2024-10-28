# 座標データのクラス

import os
import json
import torch
import random

class CoodinateData:
    def __init__(self):
        self.input_list = []
        self.output_list = []
        self.train_ratio = 0.9 # 全データのうち訓練データにする割合

        self.parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        self.node_size = len(self.parts) # 17

        self.input_dir = "src/prediction/data/input"
        input_files = [f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))]
        random.shuffle(input_files)

        self.batch_size = len(input_files)

        # 各データをまとめてバッチ化
        for i in range(self.batch_size):
            # jsonファイルを順に読み込み
            # 数字を変えれば使うデータを変えられる
            with open(os.path.join(self.input_dir, input_files[i]), "r") as json_open:
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


    def batch_size(self):
        return len(self.input_list)
    
    def input_dim(self):
        return len(self.input_list[0][0])

    def output_dim(self):
        return len(self.output_list[0][0])

    def get_train_inputs(self):
        return self.input_list[0:int(self.batch_size * self.train_ratio)]

    def get_train_outputs(self):
        return self.output_list[0:int(self.batch_size * self.train_ratio)]

    def get_test_inputs(self):
        return self.input_list[int(self.batch_size * self.train_ratio):self.batch_size]

    def get_test_outputs(self):
        return self.output_list[int(self.batch_size * self.train_ratio):self.batch_size]
