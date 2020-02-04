import matplotlib.pyplot as plt
import numpy as np
import glob
import torch

from preprocessor import Preprocessor

# few parameters
ttheta_value_number = 1037
sector_number = 1080
nb_of_samples = 75

file_path = "../data/IH-HC-519/normalized_torch_data_1080x1450"

all_data = np.zeros([0, ttheta_value_number * sector_number])

for i, data_path in enumerate(glob.glob(file_path + "/*")):
    if i < nb_of_samples:
        # data_point = Preprocessor.convert_data_to(data_path, "numpy")
        data_point = torch.load(data_path)
        data_point = data_point.numpy()
        data_point = data_point.reshape([1, ttheta_value_number*sector_number])
        all_data = np.concatenate((all_data, data_point))
all_data = all_data.reshape([ttheta_value_number*sector_number*nb_of_samples])
print(all_data.shape)
print("Computing Histogram...")
plt.hist(all_data, bins=100)
plt.show()
