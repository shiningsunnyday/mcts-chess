import sys

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from ray.rllib.models.torch.torch_modelv2 import *

from tqdm import tqdm
import os

from utils import convert_bit_mask

from games.gardner import GardnerMiniChessGame

TEST_BOARD = [[-479, -280, -320, -929, -60000], [-100, -100, -100, -100, -100], [0, 0, 0, 0, 0], [100, 100, 100, 100, 100], [479, 280, 320, 929, 60000]]


class MCGardnerNNet(nn.Module, TorchModelV2):
    def __init__(self, game, config, *args, **kwargs):
        TorchModelV2.__init__(self, *args, **kwargs)
        super(MCGardnerNNet, self).__init__()
        
        # game params
        self.game = game
        # self.board_x, self.board_y = (self.game.width, self.game.height)
        self.board_x, self.board_y = (5, 5)
        self.action_size = self.game.getActionSize()
        self.args = config

        num_channels = self.args['num_channels']
        self.num_channels = num_channels

        
        self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s, *args, **kwargs):
        
        # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y) # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))           # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))           # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))           # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))           # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args['dropout'], training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args['dropout'], training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'pool' in self_dict: del self_dict['pool']
        return self_dict

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.state_dict(),
        }, filepath)

        
    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))
        map_location = None if self.args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])

class Games(Dataset):
    def __init__(self, path_x, path_y):
        self.data = np.load(path_x)
        self.y = np.load(path_y)

    def __getitem__(self, idx):
        return self.data[idx], self.y[idx]

    def __len__(self):
        return len(self.y) 

def train(num_epochs=10):
    config = {"num_channels": 512, "dropout": 0.3}
    net = MCGardnerNNet(GardnerMiniChessGame(), config)


    path_x_train = "data/checkpoint_0_train_x.npy"
    path_y_train = "data/checkpoint_0_train_y.npy"
    path_x_test = "data/checkpoint_0_test_x.npy"
    path_y_test = "data/checkpoint_0_test_y.npy"

    g_train = Games(path_x_train, path_y_train)
    g_test = Games(path_x_test, path_y_test)
    train = DataLoader(g_train, batch_size=100, shuffle=True)
    test = DataLoader(g_test, batch_size=100)

    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)

    loss_fn = nn.MSELoss()

    best = float("inf")

    for i in range(num_epochs):
        total_test_loss = 0.0
        
        with torch.no_grad():
            for (batch, y) in tqdm(test): 
                batch, y = batch.float(), y.float()
                _, out = net.forward(batch)
                loss = loss_fn(out, y)
                total_test_loss += loss.item()

        print("Epoch %d, Test Loss: %f" % (i, total_test_loss))
        if total_test_loss < best:
            best = total_test_loss
            net.save_checkpoint(filename='epoch_%d_testloss_%f' % (i, total_test_loss))


        total_train_loss = 0.0

        for (batch, y) in tqdm(train):

            batch, y = batch.float(), y.float()
            optimizer.zero_grad()
            _, out = net.forward(batch)
            loss = loss_fn(out, y)
            if loss == loss:
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                print(loss.item())

        print("Epoch %d, Train Loss: %f" % (i, total_train_loss))

        

            





if __name__ == "__main__":
    config = {"num_channels": 512, "dropout": 0.3, "cuda": False}
    net = MCGardnerNNet(game=GardnerMiniChessGame(), config=config, obs_space=None, action_space=None, 
    num_outputs=None, model_config={}, name="")
    net.load_checkpoint(filename="epoch_7_testloss_2.808453")
    


    
    