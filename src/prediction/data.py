# 座標データのクラス

import json

class CoodinateData:
    def __init__(self, filename):
        with open(filename, "r") as json_open:
            json_load = json.load(json_open)
            self.input_list = []
            self.output_list = []
            self.input_seqsize = 60
            self.output_seqsize = 30
            self.batch_size = 10
            self.node_size = 17

            self.parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

            for i in range(self.input_seqsize):
                batch_list = []
                for j in range(self.batch_size):
                    coodinate_list = []
                    for k in range(self.node_size):
                        coodinate_list.extend(json_load[i]["keeper-pose"][self.parts[k]])
                    batch_list.append(coodinate_list)
                self.input_list.append(batch_list)

            for i in range(self.output_seqsize):
                if i + self.input_seqsize >= len(json_load):
                    break
                batch_list = []
                for j in range(self.batch_size):
                    coodinate_list = []
                    for k in range(self.node_size):
                        coodinate_list.extend(json_load[i + self.input_seqsize]["keeper-pose"][self.parts[k]])
                    batch_list.append(coodinate_list)
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

