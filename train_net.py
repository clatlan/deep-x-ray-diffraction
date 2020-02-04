import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from deepnets.mlpautoencoder import MLPAutoencoder, MLPAutoencoder_BN, ConvAutoencoder
from sectordataset import SectorDataset
from utils import compute_mean_and_std

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import dataset
dataset = SectorDataset("../data/IH-HC-519/normalized_torch_data_1080x1450/", sector_number=1080)
# dataset = SectorDataset("../data/generated_images/", sector_number=1080)
# Data loader
batch_size = 36
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=True)


# mean, std = compute_mean_and_std(loader)
# print(mean, std)

# Hyper-parameters
input_size = loader.dataset[0].size()[-1]
num_epochs = 3
learning_rate = 0.001

# network = MLPAutoencoder(input_size).to(device)
network = MLPAutoencoder_BN(input_size).to(device)

# print model parameters
for param_tensor in network.state_dict():
    print(param_tensor, "\t", network.state_dict()[param_tensor].size())

network.train() # set the model in training mode
#
# Loss and optimizer
criterion = nn.MSELoss()
# criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-5)
num_step = len(loader)

losses = []
for epoch in range(num_epochs):
    for i, vectors in enumerate(loader):
        # print(i, vectors.shape)
        vectors = vectors.to(device)
        # print(vectors.shape)
        # print(loader.dataset.data[0].view(1, -1).shape)

        # Forward pass
        pred_vectors = network.forward(vectors.float())
        loss = criterion(pred_vectors, vectors.float())

        # Backward and optimize phase
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, num_step, loss.item()))
        losses.append(loss.item())


plt.figure()
plt.plot(range(len(losses)), losses, label="train_loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.show()


# Save the model checkpoint
torch.save(network.state_dict(), '../data/IH-HC-519/test.ckpt')
print("Model has been saved")
