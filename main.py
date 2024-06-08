import os
import glob
import torch
import random
import argparse
from torchvision.transforms import v2
from torchvision.io import read_image
from torchvision.utils import save_image

from models.edsr import EDSR
from dataset import *
from utils import make_dir

parser = argparse.ArgumentParser()

### General Flags

parser.add_argument('-n', '--n', type=int, default=5, help='number of training epochs')
parser.add_argument('--seed', type=int, default=128, help='manual random seed')
parser.add_argument('--batchsize', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')

#parser.add_argument('--latent', type=int, default=512, help='size of latent dimension')

### Dataset Flags

parser.add_argument('--augment', type=str, default=None, help='augment dataset to input directory')

## Test Flags

parser.add_argument('--test', type=str, default=None, help='test model with weights from input path')

args = parser.parse_args()

if not args.test:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    print("Using cuda")
    args.device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    print("Using mps")
    args.device = torch.device("mps")
else: 
    print("Using cpu")
    args.device = torch.device("cpu")

##### Dataset #####

# hardcode to 128 by 128 images for now
# TODO make this a parameter
args.dim = 128

if args.augment:
    make_dir(args.augment)

    to_float32 = v2.ToDtype(dtype=torch.float32, scale=True)

    print("### Augmenting Dataset ###")
    counter = 0
    new_counter = 0
    for f in glob.glob("jap-art/*/*.jpg"):
        counter += 1
        img = to_float32(read_image(f).to(args.device))
        dir_name = f.split('/')[-2]
        img_name = f.split('/')[-1][:-4]
        store_location = args.augment + dir_name + "/" + img_name
        if not os.path.exists(args.augment + dir_name):
            os.makedirs(args.augment + dir_name)
        save_image(img, store_location + ".jpg")

        augment_transforms = [
            v2.RandomHorizontalFlip(p=1.0),
            v2.RandomRotation(30, fill=1),
            v2.RandomResizedCrop(720),
            v2.RandomPerspective(distortion_scale = 0.25, p=1.0, fill=1.0),
        ]

        for i, transform in enumerate(augment_transforms):
            new_counter += 1
            aug_img = transform(img)
            save_image(aug_img, store_location + f"_aug{i}.jpg")

    print(f"Original Dataset Size: {counter}")
    print(f"Augmented Dataset Size: {counter + new_counter}")
    print("#########################")

dataset = JapArtDataset(args)

# assuming channels first dataset
args.channel_size = len(dataset[0][0])

##### Model #####

model = EDSR(args, dataset)

if not args.test:
    model.train()
    model.generate(model.run_dir)
else:
    model.generate(args.test)