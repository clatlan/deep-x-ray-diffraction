from pylab import *
import numpy as np
import torch

file_path = '/home/clatlan/Documents/Projet 3A/IntegrationTests/P07-09-19/Calib_CeO2/CeO2_1500mm-00001.data'

f = open(file_path, "r")
first_line = f.readline().split()
# position_number = int(first_line[1])
position_number = 2000
sector_number = int(first_line[5])
print("Sector number is: ", sector_number)
np_data = np.zeros([sector_number, 1, position_number], dtype=float)

for i, line_number in enumerate(range(1, position_number)):
    line = f.readline().split()
    array = []
    for k, item in enumerate(line[1:]):
        np_data[k, 0, i] = float(item)


# torch_tensor = torch.from_numpy(np.data)
torch_data = torch.from_numpy(np_data)
torch_data.to(dtype=torch.float, copy=False)

# torch_data = torch.unsqueeze(torch_data, 0)
print(torch_data.size())

torch.save(torch_data, '../data/CeO2/torch_data.pt')
print('Data have been save.')
