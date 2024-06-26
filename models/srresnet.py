import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from utils import *

##### SRResNet #####

class SRResNet:

    def __init__(self, 
                args,
                dataset = None,
                ):
        
        self.args = args
        
        self.train_loader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True)

        self.channel_size = args.channel_size

        self.run_dir = "train/srresnet-" + datetime.now().strftime("%Y-%m-%d(%H:%M:%S)" + "/")
        self.progress_dir = self.run_dir + "progress/"
        make_dir(self.run_dir)
        make_dir(self.progress_dir)
        
    def train(self):
        
        model = SRResNet(self.args)
        model.apply(weights_init)
        model.to(self.args.device)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, betas=(0.5, 0.999))
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda _: 0.5)

        sample_hr, sample_lr = next(iter(self.train_loader))

        mse = nn.MSELoss()
        train_losses = []

        iters = 0
        batch_overflow = self.args.n % len(self.train_loader)
        batch_remainder = len(self.train_loader) - batch_overflow if batch_overflow > 0 else 0
        stop_iter = self.args.n + batch_remainder - 1

        print("### Begin Training Procedure ###")
        with tqdm(total=stop_iter) as pbar:
            while iters < stop_iter:
                for batch in self.train_loader:
                    batch_hr, batch_lr = batch
                    batch_hr = batch_hr.to(self.args.device)
                    batch_lr = batch_lr.to(self.args.device)                   

                    model.zero_grad()
                    batch_hat = model(batch_lr)
                    loss = mse(batch_hr, batch_hat)
                    loss.backward()
                    optimizer.step()

                    ####   Metrics Tracking  ####

                    if iters % 100 == 0:
                        train_losses.append(loss.item())
                        print(f'[%d/%d]\tloss: %.4f'
                            % (iters, self.args.n, loss.item()))

                    if (iters % 5e3 == 0) or (iters == len(self.train_loader)-1):
                        with torch.no_grad():
                            model.eval()
                            sample_hr_rec = model(sample_lr.to(self.args.device))
                            sample_hr_rec = sample_hr_rec.detach().cpu()
                            sample_lr = sample_lr.cpu()
                            model.train()
                        plot_compare_batch(sample_hr, sample_lr, sample_hr_rec, self.progress_dir + f"comp:{iters}")

                    #### Update Counters ####
                    if iters % 2e4 == 0:
                        scheduler.step()
                    iters += 1
                    pbar.update(1)

        print("### End Training Procedure ###")
        self.save_train_data(train_losses, model)

    def save_train_data(self, train_losses, model):

        # save model
        torch.save(model.state_dict(), self.run_dir + '/sr_resnet.pt')

        # save losses
        plt.figure(figsize=(10,5))
        plt.title("Model Loss")
        plt.plot(train_losses,label="train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.run_dir + "train_losses")

###############

class SuperResolutionResidualNetwork(nn.Module):
    def __init__(self, args, B = 16, F = 64):
        super(SuperResolutionResidualNetwork, self).__init__()
        self.args = args

        self.input_conv = nn.Sequential(
            nn.Conv2d(args.channel_size, F, 9, 1, 4, bias=False),
            nn.PReLU(init=0.2)
        )
        
        self.residual = nn.Sequential(
            *[ResNetBlock(F) for _ in range(B)]
        )

        self.mid_conv = nn.Sequential(
            nn.Conv2d(F, F, 3, 1, 1, bias=False),
            nn.BatchNorm2d(F)
        )

        self.upsample4x = nn.Sequential(
            nn.Conv2d(F, F * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(init=0.2),
            nn.Conv2d(F, F * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
            nn.PReLU(init=0.2),
        )

        self.output_conv = nn.Conv2d(F, args.channel_size, 9, 1, 4, bias=False)
    
    def forward(self, x):
        x = self.input_conv(x)
        res = x.clone()
        x = self.residual(x)
        x = self.mid_conv(x)
        x += res
        x = self.upsample4x(x)
        x = self.output_conv(x)
        return x
    
class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(init=0.2),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        res = x
        return res + self.model(x)
    
#########################
