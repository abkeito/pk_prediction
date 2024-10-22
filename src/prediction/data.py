# 座標データのクラス
class CodinateData:
    def __init__(self, filename):
        with open(file_name, "r") as f:
            self.input_list = []
            self.output_list = []

            # それぞれの配列に座標情報をパック。ファイルデータから読み込むかYOLO11n-poseを用いて直接データを作るかは未定

    def input_size(self):
        return len(self.input_list)
    
    def input_dim(self):
        return len(self.input_list[0])

    def output_dim(self):
        return len(self.output_list[0])

    def get_inputs(self):
        return self.input_list

    def get_outputs(self):
        return self.output_list