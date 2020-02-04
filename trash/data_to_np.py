from pylab import *
import numpy as np
import matplotlib.pyplot as plt

file_path = '/home/clatlan/Documents/Materiaux/Projet 3A/IntegrationTests/P07-09-19/Calib_CeO2/CeO2_1500mm-00001.data'

f = open(file_path, "r")
first_line = f.readline().split()
position_number = int(first_line[1])
sector_number = int(first_line[5])
data = [np.array(f.readline().split()[1:])]
np_data = np.zeros([sector_number, position_number], dtype=float)

for i, line_number in enumerate(range(1, position_number)):
    line = f.readline().split()
    array = []
    for k, item in enumerate(line[1:]):
        np_data[k, i] = float(item)
print(np_data.shape)

# print(np.amax(np_data))
# np_data = np_data / np.amax(np_data)

plt.figure(0)
# plt.imshow(np_data, interpolation='nearest', cmap=cm.jet, aspect='auto')
plt.imshow(np.array(list(map(np.log, np_data))), interpolation='nearest', cmap=cm.jet, aspect='auto')
cb = colorbar()
# clim(0.0, 500)
plt.show()


