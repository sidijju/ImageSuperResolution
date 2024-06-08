import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
from utils import find_images_in_directory

class JapArtDataset(Dataset):

    def __init__(self, args):
        self.img_dir = args.augment if args.augment else 'jap-art/'
        self.dim = args.dim
        self.img_names = []

        self.img_names = find_images_in_directory(self.img_dir + "*/")

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = read_image(self.img_names[idx])
        assert img.shape[0] <= 3

        c, w, h = img.size()
        if c < 3:
            img = torch.cat((img, img, img), dim=0)

        # resize so maximum dimension is 720
        pad = None
        if h > w:
            w = int(w * 720 / h)
            if w % 2 != 0:
                w += 1
            h = 720

            pad = [0, (720 - w)//2]
        elif w > h:
            h = int(h * 720/w)
            if h % 2 != 0:
                h += 1
            w = 720
            pad = [(720 - h)//2, 0]
        else:
            pad = 0
        
        transform = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(w, h)),
            v2.Pad(pad, fill=1),
        ])

        img = transform(img)
        lr_img = v2.Resize(self.dim)(img)
        lr_img = torch.clamp(lr_img, 0, 1)
        img = img * 2 - 1 # scale hr between -1 and 1
        
        return img, lr_img
    

class TestDataset(Dataset):
    def __init__(self, args, img_names):
        super(TestDataset, self).__init__()
        self.args = args
        self.img_names = img_names

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

        return img