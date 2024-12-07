# Transformerの入出力を管理するクラス

import os
import json
import torch
from torch.nn.utils.rnn import pad_sequence

class ClassificationData:
    def __init__(self, dirname: str):
        self.input_list = []
        self.output_list = []

        self.input_dir = dirname
        self.input_files = sorted([f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f))])
        self.input_len = len(self.input_files)

        # inputとoutputに分けて座標データをテンソルにする
        self.set_data()

        # 標準化のために入力の平均と標準偏差を求める。
        self.all_inputs_tensor = torch.cat(self.input_list, dim=0)

        self.mean = self.get_mean()
        self.std = self.get_std()

        # 標準化
        self.standardized_input_list = self.standardize(self.input_list)

        # パディングとマスクの生成
        self.input = self.pad(self.standardized_input_list)
        self.input_padding_mask = self.get_pad_mask(self.standardized_input_list)

    # inputとoutputに分けて座標データをテンソルにする
    def set_data(self) -> None:

        for i in range(self.input_len):
            # jsonファイルを順に読み込み
            with open(os.path.join(self.input_dir, self.input_files[i]), "r") as json_open:
                json_load = json.load(json_open)
                input_frames_list = json_load["input"]
                output_labels_list = json_load["label"]
                # 座標情報をテンソルにしてからリストに追加
                input_frames_tensor = torch.tensor(input_frames_list, dtype=torch.float32)
                output_frames_tensor = torch.tensor(output_labels_list, dtype=torch.float32)
                # テンソルが空でないことを確認
                if input_frames_tensor.size(0) > 0 and output_frames_tensor.size(0) > 0:
                    self.input_list.append(input_frames_tensor)
                    self.output_list.append(output_frames_tensor)

    # 各関節座標ごとの平均
    def get_mean(self) -> torch.Tensor:
        return self.all_inputs_tensor.mean(dim=0)
    
    # 各関節座標ごとの標準偏差
    def get_std(self) -> torch.Tensor:
        std = self.all_inputs_tensor.std(dim=0)
        # 0除算対策
        std_epsilon = 10**-9
        if (std < std_epsilon).any():
            std = torch.full_like(std, std_epsilon)
        return std
    

    # 正規化
    def standardize(self, tensor_list: list) -> list:
        res = []
        for frames_tensor in tensor_list:
            standardized_tensor = (frames_tensor - self.mean) / self.std
            res.append(standardized_tensor)
        # 標準化後のテンソルを返す
        return res
    
    def destandardize(self, output: torch.Tensor) -> list:
        return output * self.std + self.mean
    
    # 足りないフレームを0埋めして系列長を揃える。リストをテンソルへ
    def pad(self, tensor_list: list) -> torch.Tensor:
        padded_tensors = pad_sequence(tensor_list, batch_first=False, padding_value=0.)
        return padded_tensors
    
    # 上で埋めたフレームの部分がFalse, それ以外がTrueとなるマスクを作成
    def get_pad_mask(self, tensor_list: list) -> torch.Tensor: 
        masks = [torch.ones_like(tensor, dtype=torch.bool) for tensor in tensor_list]
        padded_masks = pad_sequence(masks, batch_first=False, padding_value=False)
        return padded_masks
    
    # モデルの入力
    def get_input(self) -> torch.Tensor:
        return self.input # 時系列 * データ数 * 次元数

    # 入力用のpadding mask
    def get_input_padding_mask(self) -> torch.Tensor:
        # paddingした所のみをTrueにするので反転させ、さらに[batch_size, seq_len]の形状にするために転置
        input_padding_mask = ~self.input_padding_mask.permute(2, 1, 0)[0]
        return input_padding_mask
    
    
    # モデルの出力（ラベル）
    def get_target(self) -> torch.Tensor:
        return torch.stack(self.output_list).unsqueeze(0)

    # モデルの入力の各次元がわかる
    def get_input_dim(self) -> int:
        return self.input.shape[2]
    
    def get_output_dim(self) -> int:
        return self.get_target().shape[2]
    
    def get_input_seq_len(self) -> int:
        return self.input.shape[0]
    
    def get_output_seq_len(self) -> int:
        return self.get_target().shape[0]
    
    # ミニバッチを作成
    def batchify(self, data: torch.Tensor, mask: torch.Tensor, batch_size: int) -> tuple:
        original_batch_size = data.shape[1]
        # batch間でシャッフルを行う
        random_index = torch.randperm(original_batch_size)
        shuffled_data = data[:, random_index, :]

        # ミニバッチの塊がいくつあるか
        num_minibatch = original_batch_size // batch_size
        batchified_data_list = []

        # ミニバッチに分割しリストに追加
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