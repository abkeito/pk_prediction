import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# Load a subset of the MNIST dataset
def load_data(json_filepaths):
    dataset = []
    for filepath in json_filepaths:
        with open(filepath, "r") as f:
            dataset.append(json.load(f))
    # Load dataset
    X = [data["input"] for data in dataset] # 52 * 1 * (可変長) がたくさんある
    Y = [data["label"] for data in dataset] # 8 * 1 がたくさんある
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.9)

    # Calculate the subset size for training and testing data
    print(len(X_train), len(Y_train), len(X_valid), len(Y_valid))
    # Return the subset of the dataset
    return X_train, X_valid, Y_train, Y_valid

class VariableLengthDataset(Dataset):
    def __init__(self, data, labels):
        self.data = [torch.tensor(d, dtype=torch.float32) for d in data]
        self.labels = [torch.tensor(l, dtype=torch.float32) for l in labels]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def collate_fn(batch):
    data, labels = zip(*batch)
    
    # 各系列の長さを取得
    lengths = torch.tensor([len(seq) for seq in data])

    # 系列データにパディングを追加
    padded_data = pad_sequence(data, batch_first=True, padding_value=0)

    # ラベルも同様に扱う（ここでは長さを固定と仮定）
    labels = torch.stack(labels)

    # LSTMの入力形式 (シーケンス長, バッチサイズ, 特徴数) へ変換
    # padded_data = padded_data.permute(1, 0, 2)  # (バッチサイズ, シーケンス長, 特徴数) -> (シーケンス長, バッチサイズ, 特徴数)

    return padded_data, labels

def preprocess_data(data, labels, batch_size=1):
    dataset = VariableLengthDataset(data, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader

def get_edge_matrix():
    original_edge_list = [[0,1], [0,2],[1,2], [1,3], [3,5], [2,4], [4,6], [1,7], [2,8], [7,8], [7,9], [9,11], [8,10], [10, 12]]
    edge_list = [[0,1], [0,2],[1,2], [1,3], [3,5], [2,4], [4,6], [1,7], [2,8], [7,8], [7,9], [9,11], [8,10], [10, 12]]
    # エッジの対応部で重み付けを入れるだけ
    for edge in original_edge_list:
        id_0, id_1 = edge[0], edge[1]
        edge_list.append([id_1, id_0])
        edge_list.append([2*id_0, 2*id_1])
        edge_list.append([2*id_1, 2*id_0])
        edge_list.append([2*id_0+1, 2*id_1+1])
        edge_list.append([2*id_1+1, 2*id_0+1])
        edge_list.append([2*id_0+26, 2*id_1+26])
        edge_list.append([2*id_1+26, 2*id_0+26])
        edge_list.append([2*id_0+27, 2*id_1+27])
        edge_list.append([2*id_1+27, 2*id_0+27])
    for i in range(52):
        edge_list.append([i,i])
    
    return edge_list