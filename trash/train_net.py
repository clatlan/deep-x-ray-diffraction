import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

from deepnets.mlpautoencoder import MLPAutoencoder, MLPAutoencoder_Leaky, ConvAutoencoder
from read_image import display_image_from_torch_tensor


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# import dataset
dataset = torch.load('../data/CeO2/torch_data.pt')

# Data loader
batch_size = 36
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     shuffle=False)

# print(loader.dataset.data[0].shape)

# Hyper-parameters
input_size = 2000
num_epochs = 1
learning_rate = 0.0001

network = ConvAutoencoder().to(device)
# print model parameters
for param_tensor in network.state_dict():
    print(param_tensor, "\t", network.state_dict()[param_tensor].size())
network.train() # set the model in training mode

# Loss and optimizer
# criterion = nn.MSELoss()
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-5)
num_step = len(loader)

for epoch in range(num_epochs):
    for i, vectors in enumerate(loader):
        # print(i, vectors.shape)
        vectors = vectors.to(device)
        # print(vectors.shape)
        # print(loader.dataset.data[0].view(1, -1).shape)

        # Forward pass
        pred_vectors = network(vectors.float())
        loss = criterion(pred_vectors, vectors.float())

        # Backward and optimize phase
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if (i + 1) % 5 == 0:
        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, i + 1, num_step, loss.item()))

# Save the model checkpoint
torch.save(network.state_dict(), 'first_network.ckpt')
print("Model has been saved")


network.eval()

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
source_images = torch.zeros((0, 1, 2000), dtype=torch.double)
output_images = torch.zeros((0, 1, 2000), dtype=torch.double)
with torch.no_grad():
    for i, vectors in enumerate(loader):
        vectors = vectors.to(device)
        # images = images.view(images.size(0), -1)

        outputs = network(vectors.float())

        # print(images.shape, outputs.shape)

        sample_vector = vectors[0]
        sample_pred_vector = outputs[0]
        # print(sample_vector.shape, sample_pred_vector.shape)

        # source_images.append(sample_vector)
        # output_images.append(sample_pred_vector)

        # sample_image = sample_image.view(28, 28)
        # sample_pred_image = sample_pred_image.view(28, 28)

        # print('shape of sample: ', sample_vector.numpy().shape)
        # plt.imsave(os.path.join(in_dir, 'sample_%d.png' % i), sample_image.numpy())
        # plt.imsave(os.path.join(out_dir, 'sample_%d.png' % i), sample_pred_image.numpy())

        # imgplot = plt.imshow(sample_image.numpy())

        # concatenate vectors to remake image
        source_images = torch.cat((source_images, vectors.double()))
        output_images = torch.cat((output_images, outputs.double()))
source_images = source_images.squeeze()
output_images = output_images.squeeze()
print("source image shape: ", source_images.size())
print("source image shape: ", output_images.size())



# torch.save(source_images, '../data/source_image.pt')
# torch.save(output_images, '../data/output_image.pt')
#
# # display_image_from_torch_tensor('../data/source_image.pt', title="Source Image")
# # display_image_from_torch_tensor('../data/output_image.pt', title="Output Image")
# #
