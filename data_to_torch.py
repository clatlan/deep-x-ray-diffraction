import torch
import glob
import os
import matplotlib.pyplot as plt

from preprocessor import Preprocessor


if __name__ == "__main__":

    files_path = "../data/IH-HC-519/integrated_data_1080x1450/"

    for i, data_point_path in enumerate(glob.glob(files_path + "/*")):
        torch_point = Preprocessor.convert_data_to(data_point_path, "torch", normalize=True)

        torch.save(torch_point,
                   "../data/IH-HC-519/normalized_torch_data_1080x1450/" + data_point_path.split(os.path.sep)[-1][:-5] + ".pt")


        # # if i < 10:
        #     numpy_point = Preprocessor.convert_data_to(data_point_path, "numpy")
        #     print(torch_point.size())
        #     print(numpy_point.shape)
        #     # numpy_point = torch_point.view(1080, 1450).numpy()
        #     # plt.matshow(numpy_point)
        #     # plt.show()
        # # print(data_point_path.split(os.path.sep)[-1][:-5])
        # torch.save(torch_point,
        #            "../data/IH-HC-519/torch_data_1080x1450/" + data_point_path.split(os.path.sep)[-1][:-5] + ".pt")
