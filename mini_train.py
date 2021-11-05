import sys

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import convert_bit_mask

from games.gardner import GardnerMiniChessGame

TEST_BOARD = [[-479, -280, -320, -929, -60000], [-100, -100, -100, -100, -100], [0, 0, 0, 0, 0], [100, 100, 100, 100, 100], [479, 280, 320, 929, 60000]]


class MCGardnerNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.game = game
        # self.board_x, self.board_y = (self.game.width, self.game.height)
        self.board_x, self.board_y = (5, 5)
        self.action_size = self.game.getActionSize()
        self.args = args

        num_channels = args['num_channels']
        self.num_channels = num_channels

        super(MCGardnerNNet, self).__init__()
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

    def forward(self, s):
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



if __name__ == "__main__":
    config = {"num_channels": 512, "dropout": 0.3}
    net = MCGardnerNNet(GardnerMiniChessGame(), config)


    data = np.load("data/checkpoint_0_train_x.npy")
    y = np.load("data/checkpoint_0_train_y.npy")

    x = torch.FloatTensor(data[:32])
    y = torch.FloatTensor(y[:32])

    print(x.shape, y.shape)

    _, v = net.forward(x)
    print(v)
    print(y)
    loss = nn.MSELoss()(v, y)
    print(loss)
    


    
    