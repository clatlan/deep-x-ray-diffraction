import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm

file_path = 'Al_powder_001.data'
f = open(file_path, "r")

first_line = f.readline().split() # first line is read
tth_number = int(first_line[1])
sector_number = int(first_line[5])
print("tth_number is: ", tth_number)
print("Sector number is: ", sector_number)

np_data = np.zeros([sector_number, 1, tth_number], dtype=float)
image = np.zeros([sector_number, tth_number], dtype=float)

for line_number in range(tth_number):
    line = f.readline().split()[1:]
    for k, item in enumerate(line):
        np_data[k, 0, line_number] = float(item)
        image[k, line_number] = float(item)
f.close()

ys, xs = np.where(image != 0)
print(xs, ys)
image = image[:,:max(xs)+1]
np_data = np_data[:, :, :max(xs)+1]

# image = image[np.all(image == 0 and np.where(image==0, image) > 1000, axis = 0)]
# print(image)

torch_data = torch.from_numpy(np_data)
torch_data.to(dtype=torch.float, copy=False)

print("torch data shape: ", torch_data.size())
print("image shape: ", image.shape)

plt.figure()
plt.imshow(image, interpolation='nearest', cmap=cm.jet, aspect="auto")
cb = plt.colorbar()
plt.clim(-1.20, 200)
plt.show()
