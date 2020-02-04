import torch
import os
import matplotlib.pyplot as plt

from read_image import display_image_from_torch_tensor
from deepnets.mlpautoencoder import MLPAutoencoder, MLPAutoencoder_Leaky, ConvAutoencoder

# make directories of input and original images if not existing
in_dir = './mlp_autoencoder_orignal/'
out_dir = './mlp_autoencoder_output/'

if not os.path.exists(in_dir):
    os.mkdir(in_dir)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


working_dir = "../data/CeO2/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model parameters
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("first_network.ckpt"))
model.eval()


# import dataset
dataset = torch.load(working_dir + "torch_data.pt")
loader = torch.utils.data.DataLoader(dataset=dataset,
                                     shuffle=False)

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
source_images = torch.zeros((0, 1, 2000), dtype=torch.double)
output_images = torch.zeros((0, 1, 2000), dtype=torch.double)
with torch.no_grad():
    for i, vectors in enumerate(loader):
        vectors = vectors.to(device)

        outputs = model(vectors.float())

        # plt.imsave(os.path.join(in_dir, 'sample_%d.png' % i), sample_image.numpy())
        # plt.imsave(os.path.join(out_dir, 'sample_%d.png' % i), sample_pred_image.numpy())

        # imgplot = plt.imshow(sample_image.numpy())

        # concatenate vectors to remake image
        source_images = torch.cat((source_images, vectors.double()))
        output_images = torch.cat((output_images, outputs.double()))


source_images = source_images.squeeze()
output_images = output_images.squeeze()

print("source image shape: ", source_images.shape)
print("source image shape: ", output_images.shape)

torch.save(source_images, working_dir + "images/source_image.pt")
torch.save(output_images, working_dir + "images/output_image.pt")

plt.imsave(working_dir + "images/np_source_image.png", source_images.numpy())
plt.imsave(working_dir + "images/np_output_image.png", output_images.numpy())

display_image_from_torch_tensor(working_dir + "images/source_image.pt", title="Source Image", save_path=working_dir+"images/source_image.png")
display_image_from_torch_tensor(working_dir + "images/output_image.pt", title="Output Image", save_path=working_dir+"images/output_image.png")
