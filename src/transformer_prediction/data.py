# 座標データのクラス

import os
import json
import torch
from torch.nn.utils.rnn import pad_sequence

class CoodinateData:
    def __init__(self, dirname: str):
        self.input_list = []
        self.output_list = []

        self.parts = ['face', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'face-move', 'left_shoulder-move', 'right_shoulder-move', 'left_elbow-move', 'right_elbow-move', 'left_wrist-move', 'right_wrist-move', 'left_hip-move', 'right_hip-move', 'left_knee-move', 'right_knee-move', 'left_ankle-move', 'right_ankle-move']
        self.node_size = len(self.parts)

        self.input_dir = dirname
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))])

        self.input_len = len(self.input_files)

        self.set_data()

        # 標準化のために平均と標準偏差を求める
        all_inputs_tensor = torch.cat(self.input_list, dim=0)
        all_outputs_tensor = torch.cat(self.output_list, dim=0)
        self.all_tensor = torch.cat((all_inputs_tensor, all_outputs_tensor), dim=0)

        self.mean = self.get_mean()
        self.std = self.get_std()

        self.standardized_input_list = self.standardize(self.input_list)
        self.standardized_output_list = self.standardize(self.output_list)

        self.input = self.pad(self.standardized_input_list)
        self.input_padding_mask = self.get_pad_mask(self.standardized_input_list)
        self.output = self.pad(self.standardized_output_list)
        self.output_padding_mask = self.get_pad_mask(self.standardized_output_list)


    # inputとoutputに分けて座標データをテンソルにする
    def set_data(self) -> None:
        goal_center = [3.66, 1.22] # ゴールの中心座標
        
        for i in range(self.input_len):
            # jsonファイルを順に読み込み
            with open(os.path.join(self.input_dir, self.input_files[i]), "r") as json_open:
                json_load = json.load(json_open)
                input_frames_list = []
                output_frames_list = []
                # フレームごとに座標の情報をリストに格納
                for frame in json_load:
                    data_type = frame["data_type"]
                    if data_type is None:
                        continue
                    keeper_pose = frame["keeper-pose"]
                    if keeper_pose is None:
                        continue
                    coodinates_list = []
                    for j in range(self.node_size):
                        # 読み取れなかったノードはゴールの中心にあると仮定
                        if keeper_pose[self.parts[j]] == [0., 0.]:
                            coodinates_list.extend(goal_center) # ゴールの中心
                        else:
                            coodinates_list.extend(keeper_pose[self.parts[j]])
                    if data_type == "input":
                        input_frames_list.append(coodinates_list)
                    elif data_type == "output":
                        output_frames_list.append(coodinates_list)
                # 座標情報をテンソルにしてからリストに追加
                input_frames_tensor = torch.tensor(input_frames_list, dtype=torch.float32)
                output_frames_tensor = torch.tensor(output_frames_list, dtype=torch.float32)
                # テンソルが空でないことを確認
                if input_frames_tensor.size(0) > 0 and output_frames_tensor.size(0) > 0:
                    self.input_list.append(input_frames_tensor)
                    self.output_list.append(output_frames_tensor)

    # 各関節座標ごとの平均
    def get_mean(self) -> torch.Tensor:
        return self.all_tensor.mean(dim=0)
    
    # 各関節座標ごとの標準偏差
    def get_std(self) -> torch.Tensor:
        std = self.all_tensor.std(dim=0)
        # 0除算対策
        std_epsilon = 10**-9
        if (std < std_epsilon).any():
            std = torch.full_like(std, std_epsilon)
        return std
    
    def standardize(self, tensor_list: list) -> list:
        res = []
        for frames_tensor in tensor_list:
            standardized_tensor = (frames_tensor - self.mean) / self.std
            res.append(standardized_tensor)
        # 標準化後のテンソルを返す
        return res

    def pad(self, tensor_list: list) -> torch.Tensor:
        padded_tensors = pad_sequence(tensor_list, batch_first=False, padding_value=0.)
        return padded_tensors
    
    def get_pad_mask(self, tensor_list: list) -> torch.Tensor: 
        masks = [torch.ones_like(tensor, dtype=torch.bool) for tensor in tensor_list]
        padded_masks = pad_sequence(masks, batch_first=False, padding_value=False)
        return padded_masks
    
    
    def get_input(self) -> torch.Tensor:
        return self.input
    
    def get_dec_input(self) -> torch.Tensor:
        return self.output[:-1]
    
    def get_target(self) -> torch.Tensor:
        return self.output[1:] # dec_inputを右に一つずらす
    
    # Encoder用のpadding mask
    def get_input_padding_mask(self) -> torch.Tensor:
        # paddingした所のみをTrueにするので反転させ、さらに[batch_size, seq_len]の形状にするために転置
        input_padding_mask = ~self.input_padding_mask.permute(2, 1, 0)[0]
        return input_padding_mask
    
    def get_dec_input_padding_mask(self) -> torch.Tensor:
        output_padding_mask = ~self.output_padding_mask.permute(2, 1, 0)[0]
        return output_padding_mask[:, :-1]

    def get_input_dim(self) -> int:
        return self.input.shape[2]
    
    def get_output_dim(self) -> int:
        return self.output.shape[2]
    
    def get_input_seq_len(self) -> int:
        return self.input.shape[0]
    
    def get_output_seq_len(self) -> int:
        return self.output.shape[0]

    def batchify(self, data: torch.Tensor, mask: torch.Tensor, batch_size: int) -> tuple:
        original_batch_size = data.shape[1]
        # batch間でシャッフルを行う
        random_index = torch.randperm(original_batch_size)
        shuffled_data = data[:, random_index, :]

        # ミニバッチの塊がいくつあるか
        num_minibatch = original_batch_size // batch_size
        batchified_data_list = []
        for i in range(num_minibatch):
            batchified_data_list.append(shuffled_data[:, i * batch_size: (i + 1) * batch_size, :])

        if mask is None:
            return batchified_data_list

        else:
            shuffled_mask = mask[random_index]
            batchified_mask_list = []
            for i in range(num_minibatch):
                batchified_mask_list.append(shuffled_mask[i * batch_size: (i + 1) * batch_size])

        return batchified_data_list, batchified_mask_list