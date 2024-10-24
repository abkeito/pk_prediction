# 座標データのクラス
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

            for i in range(self.input_seqsize):
                batch_list = []
                for j in range(self.batch_size):
                    coodinate_list = []


            

    def batch_size(self):
        return len(self.input_list[0])
    
    def input_dim(self):
        return len(self.input_list[0][0])

    def output_dim(self):
        return len(self.output_list[0][0])

    def get_inputs(self):
        return self.input_list

    def get_outputs(self):
        return self.output_list


    