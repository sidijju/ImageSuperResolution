import torch
import argparse
from torch.utils.data import DataLoader

from models.edsr import EnhancedDeepResidualNetwork
from models.srresnet import SuperResolutionResidualNetwork
from dataset import TestDataset

from utils import plot_batch, plot_image

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='path to model weights', required=True)
parser.add_argument('--images', type=str, default=None, help='path to input image directory', required=True)

### Model Flags

parser.add_argument('--srrn', action='store_true', help='use SRResNet model')
args = parser.parse_args()

if torch.cuda.is_available():
    print("Using cuda")
    args.device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    print("Using mps")
    args.device = torch.device("mps")
else: 
    print("Using cpu")
    args.device = torch.device("cpu")

test_dataset = TestDataset(args)

# hardcode to 128 by 128 images for now
# TODO make this a parameter
args.dim = 128
args.channel_size = len(test_dataset[0][0])

if args.srrn:
    model = SuperResolutionResidualNetwork(args)
else:
    EnhancedDeepResidualNetwork(args)

model.load_state_dict(torch.load(args.model))
model.to(args.device)
model.eval()

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("### Begin Single Image Super Resolution ###")
for i, batch in enumerate(test_loader, 0):
    batch, names = batch.to(args.device)
    batch_rec = model(batch)

    for idx in range(len(batch_rec)):
        plot_image(batch_rec[idx], "rec_" + names[idx])

    plot_batch(batch, args.images + f"/x_{i}")
    plot_batch(batch_rec, args.images + f"/yhat_{i}")
print("### End Single Image Super Resolution ###")