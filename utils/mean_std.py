import torch
import numpy as np

def compute_mean_and_std(loader):
    """Compute the mean and std"""
    mean_list = []
    std_list = []

    for data in loader:

        if torch.is_tensor(data):
            data = data.numpy()

        batch_mean = np.mean(data, axis=(2))
        batch_std = np.std(data, axis=(2))

        mean_list.append(batch_mean)
        std_list.append(batch_std)

    mean = np.array(mean_list).mean(axis=0)
    std = np.array(std_list).mean(axis=0)

    print(mean, std)

    return mean, std
