# train
import sys
import torch
from torch import optim

import data
from prediction_model import CodinatePredictionModel

EPOCH_NUM = 10 # 適宜変えてね

dataset = data.CodinateData() # 引数要確認

input_size = dataset.input_dim()
output_size = dataset.output_dim()

model = CodinatePredictionModel(input_size, output_size)

optimizer = optim.Adam(model.parameters())


for epoch in range(EPOCH_NUM):
    print("{0} / {1} epoch start.".format(epoch + 1, EPOCH_NUM))

    sum_loss = 0.0
    for i, (input, output) in enumurate(zip(dataset.get_inputs(), dataset.get_outpus())):

        model.reset_state()
        optimizer.zero_grad()

        input = torch.tensor(input, dtype=torch.float).unsqueeze(-1)
        output = torch.tensor(output, dtype=torch.float).unsqueeze(-1)
        loss = model(input, output)
        loss.backward()
        optimizer.step()
        sum_loss += float(loss.data.to('cpu'))

    print("mean loss = {0}.".format(sum_loss / dataset.input_size()))

    model_file = "trained_model/prediction_" + str(epoch + 1) + ".model"
    torch.save(model.state_dict(), model_file)