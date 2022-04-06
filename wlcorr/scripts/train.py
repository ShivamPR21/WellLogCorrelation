import argparse
import os
from ast import arg
from pickletools import optimize
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ..dataset import EncoderDecoderStaticDataset
from ..models import EncoderDecoder1DCNN


def compute_loss(dl:DataLoader, model:Callable, crt:Callable):
    total_loss = 0.
    cnt = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dl:
            data = data.to(device)

            # calculate outputs by running images through the network
            output = model(data)

            loss = crt(output, data)

            # print(predicted, labels)
            total_loss += loss.item()*data.size(0)
            cnt += data.size(0)
    return total_loss/cnt-1.

class VRLoss(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.cosinesim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, output:torch.Tensor, target:torch.Tensor):
        sim = self.cosinesim(output, target)

        l = sim.square().exp().mean()
        return l

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", action='store', help="Data root directory, with train.csv and test.csv .")
    parser.add_argument("--in_channels", action='store', default=2, type=int, help="Number of channel/features to encode.")
    parser.add_argument("--encoding_len", action='store', default=10, type=int, help="Encoding length.")
    parser.add_argument("--use_gpu", action="store_true", help="If given, then GPU will be used if available.")
    parser.add_argument("--batch_size", action='store', default=20, type=int)
    parser.add_argument("--epochs", action='store', default=100, type=int)
    parser.add_argument("--iterations", action='store', default=100, type=int)
    parser.add_argument("--lr", action='store', default=0.0007, type=float)
    parser.add_argument("--Wdeacy", action='store', default=0.01, type=float)
    parser.add_argument("--save_dict", action='store', default=os.path.join(os.getenv('HOME', default="~"), 'basemodel'), type=str)
    args = parser.parse_args()

    dataset = EncoderDecoderStaticDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size = 10, shuffle = True)
    data_len = len(dataloader)
    log_n = int(data_len//3)

    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    model = EncoderDecoder1DCNN(args.in_channels, args.encoding_len)
    model.to(device)
    model.train()

    criterion = VRLoss() # Loss function
    params_list = model.parameters() # model parameters
    optimizer = optim.AdamW(params_list, lr = args.lr, weight_decay=args.Wdecay)

    for epoch in range(args.epochs):

        running_loss = 0.0
        for i, data in enumerate(dataloader):
            data = data.to(device) # Move data to target device

            # zero the parameter gradients
            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i%log_n == log_n-1:
                print(f'Epoch : {epoch}, Iteration : {i},Running loss : {running_loss}')
                running_loss = 0

    torch.save(model.state_dict(), args.save_dict)
