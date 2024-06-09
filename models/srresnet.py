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

        mse = nn.MSELoss()

        sample_hr, sample_lr = next(iter(self.train_loader))
        plot_batch(sample_lr, self.progress_dir + f"x:0")
        plot_batch(sample_hr, self.progress_dir + f"y:0")

        train_losses = []
        iters = 0

        print("### Begin Training Procedure ###")
        for epoch in tqdm(range(self.args.n)):
            for i, batch in enumerate(self.train_loader, 0):
                batch_hr, batch_lr = batch
                batch_hr = batch_hr.to(self.args.device)
                batch_lr = batch_lr.to(self.args.device)                    

                model.zero_grad()
                batch_hat = model(batch_lr)
                loss = mse(batch_hat, batch_hr)
                loss.backward()
                optimizer.step()

                #############################
                ####   Metrics Tracking  ####
                #############################

                if i % 100 == 0:
                    train_losses.append(loss.item())

                    print(f'[%d/%d][%d/%d]\tloss: %.4f'
                        % (epoch, self.args.n, i, len(self.train_loader), loss.item()))

                if (iters % 5000 == 0) or ((epoch == self.args.n-1) and (i == len(self.train_loader)-1)):

                    with torch.no_grad():
                        batch_yhat = model(sample_lr).detach().cpu()
                    plot_batch(batch_yhat, self.progress_dir + f"yhat:{iters}")

                iters += 1

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
