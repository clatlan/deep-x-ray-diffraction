import numpy as np
import torch
import glob
import os
from sklearn.preprocessing import MinMaxScaler


class Preprocessor:
    @staticmethod
    def convert_data_to(file_path, out, normalize=False, standardize=False, set_negative_to_zero=False):
        print("Working on : {}".format(file_path.split(os.path.sep)[-1]))
        f = open(file_path, "r")
        first_line = f.readline().split()
        ttheta_value_number = int (first_line[1])
        sector_number = int (first_line[5])

        data = np.zeros([sector_number, ttheta_value_number], dtype=float)

        for i in range(1, ttheta_value_number):
            line = f.readline().split()
            for k, item in enumerate(line[1:]):
                if set_negative_to_zero and float(item) < 0:
                    item = 0
                data[k, i] = float(item)
        f.close()
        ys, xs = np.where(data != 0)
        data = data[ :, :max(xs) + 1]

        if standardize:
            mean, std = Preprocessor._compute_mean_and_std(data)
            data = (data - mean)/std

        if normalize:
            data = Preprocessor._normalize(data)

        if out == "numpy":
            return data

        elif out == "torch":
            data = torch.from_numpy((data))
            # data = data.view(sector_number, 1, data.size()[-1])
            data.to(dtype=torch.float, copy=False)
            return data

    @staticmethod
    def _compute_mean_and_std(data):
        """Compute the mean and std"""
        mean = np.mean(data)
        std = np.std(data)
        return mean, std

    @staticmethod
    def _normalize(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)


if __name__ == "__main__":
    # preprocessor = Preprocessor()

    file_path = "../data/IH-HC-519/integrated_data_1080x1450/"

    for data_point_path in glob.glob(file_path + "/*"):
        torch_point = Preprocessor.convert_data_to(data_point_path, "torch")
        print(torch_point.shape)
