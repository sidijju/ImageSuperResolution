import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
from utils import find_images_in_directory

class JapArtDataset(Dataset):

    def __init__(self, args):
        self.dim = args.dim
        self.img_dir = args.augment if args.augment else 'jap-art/'
        self.img_names = find_images_in_directory(self.img_dir + "*/")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = read_image(self.img_names[idx])
        assert img.shape[0] <= 3

        c, _, _ = img.size()
        if c < 3:
            img = torch.cat((img, img, img), dim=0)
        
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(512, 512)),
        ])

        img = transform(img)
        lr_img = v2.Resize(self.dim)(img)
        lr_img = torch.clamp(lr_img, 0, 1)
        img = img * 2 - 1 # scale hr between -1 and 1

        return img, lr_img
    

class TestDataset(Dataset):
    def __init__(self, args):
        super(TestDataset, self).__init__()
        self.args = args
        self.img_names = find_images_in_directory(args.images)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = read_image(self.img_names[idx])

        assert img.shape[0] <= 3

        c, w, h = img.size()
        if c < 3:
            img = torch.cat((img, img, img), dim=0)

        assert w == self.args.dim and h == self.args.dim

        img = self.img_list[idx]
        img = v2.ToDtype(torch.float32, scale=True)(img)

        return img, self.img_names[idx]