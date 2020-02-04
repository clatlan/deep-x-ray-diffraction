import os
import torch
import numpy as np
from torch.utils.data import Dataset

class SectorDataset(Dataset):

    def __init__(self, file_dir, sector_number=1080, transform=None):
        self.file_dir = file_dir
        self.sector_number = sector_number
        self.filenames = os.listdir(file_dir)
        self.transform = transform

    def __len__(self):
        return self.sector_number * len(self.filenames)

    def __getitem__(self, index):
        file_index = int(index / self.sector_number)
        sector_index = index % self.sector_number

        file_name = self.filenames[file_index]
        file_path = self.file_dir + file_name

        if file_name[-3: ] == ".pt":
            item = torch.load(file_path)[sector_index]
        elif file_name[-4: ] == ".npy":
            item = torch.Tensor(np.load(file_path)[sector_index])

        if self.transform:
            item = self.transform(item)
        return item

    def _find_files(self, directory, pattern="*.jpg"):
        files = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
        return files
