import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import sys


def display_image_from_torch_tensor(file_path, title="No title", save_path=None):
    torch_data = torch.load(file_path)

    np_data = torch_data.numpy()
    plt.figure(0)
    plt.imshow(np_data, interpolation='nearest', cmap=cm.jet, aspect='auto')
    # plt.imshow(np.array(list(map(np.log, np_data))), interpolation='nearest', cmap=cm.jet, aspect='auto')
    cb = plt.colorbar()
    plt.clim(-1.20, 2000)
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


if "__name__" == "__main__":
    file_path = sys.argv[0]
    display_image_from_torch_tensor(file_path)
