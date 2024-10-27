# trainが動くかの検証用

import random

class RandomCoordinateData:
    def __init__(self):
        self.input_list = self.pack(40)
        self.output_list = self.pack(30)

    def pack(self, seq_size, batch_size=50, node_size=17):
        # 初期化。入出力はseq_size個の配列からなり、各batchはそれぞれ各ノードの座標を格納したbatch_size個の配列からなる。
        # 座標の時系列情報は[node1のx座標, node1のy座標, node2のx座標, node2のy座標, ... , node17のx座標, node17のy座標]という形式
        empty_list = []
        for _ in range(seq_size):
            seq_list = []
            for _ in range(batch_size):
                batch_list = []
                for _ in range(node_size):
                    batch_list.append(random.randint(0, 1920)) # x座標
                    batch_list.append(random.randint(0, 1080)) # y座標
                seq_list.append(batch_list)
            empty_list.append(seq_list)

        return empty_list

    def batch_size(self):
        return len(self.input_list[0])
    
    def input_dim(self):
        return len(self.input_list[0][0]) # 17 nodes * 2 (x, y) = 34

    def output_dim(self):
        return len(self.output_list[0][0]) # 34

    def get_inputs(self):
        return self.input_list

    def get_outputs(self):
        return self.output_list