# train
import sys
import torch
from torch import optim
from torch import cuda

# import data
import random_data
from model import CodinatePredictionModel

EPOCH_NUM = 10 # 適宜変えてね

# dataset = data.CodinateData() # 引数要確認
dataset = random_data.RandomCodinateData()

input_size = dataset.input_dim()
output_size = dataset.output_dim()

# デバイスの設定
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = CodinatePredictionModel(input_size, output_size).to(device)

optimizer = optim.Adam(model.parameters())


for epoch in range(EPOCH_NUM):
    print("{0} / {1} epoch start.".format(epoch + 1, EPOCH_NUM))

    model.reset_state()
    optimizer.zero_grad()

    inputs = torch.tensor(dataset.get_inputs(), dtype = torch.float32)
    outputs = torch.tensor(dataset.get_outputs(), dtype = torch.float32)
    loss = model(inputs, outputs)
    loss.backward()
    optimizer.step()
    sum_loss = float(loss.data.to('cpu'))

    print("mean loss = {0}.".format(sum_loss / dataset.input_size()))

    model_file = "trained_model/prediction_" + str(epoch + 1) + ".model"
    torch.save(model.state_dict(), model_file)