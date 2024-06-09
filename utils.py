import os
import torch
import torch.nn as nn
import glob
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract(x):
    for _ in range(4 - len(x.shape)):
        x = torch.unsqueeze(x, -1)
    return x

def find_images_in_directory(path):
    img_names = []
    for f in glob.glob(path + "*.jpg"):
        img_names.append(f)
    return img_names

def plot_image(image, path):
    plt.cla()
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.savefig(path)

def plot_batch(batch, path):
    plt.cla()
    grid = vutils.make_grid(batch.cpu()[:25], nrow = 5, padding=2, normalize=True)
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.savefig(path)

# utility function to iterate through model
# and initalize weights in layers rom N(0, 0.02)
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
    elif classname.find('ConvTranspose2d') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.constant_(model.bias.data, 0)