import sys

import argparse

   
from abc import ABC
import numpy as np

from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.sample_batch import SampleBatch

from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

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

class MCGardnerNNetTrain(nn.Module):
    def __init__(self):

        super(MCGardnerNNetTrain, self).__init__()
        
        # game params
        self.game = GardnerMiniChessGame()
        # self.board_x, self.board_y = (self.game.width, self.game.height)
        self.board_x, self.board_y = (5, 5)
        self.action_size = self.game.getActionSize()


        num_channels = 512
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



    def forward(self, s):

        # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y) # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)) if s.shape[0] > 1 else self.conv1(s))           # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)) if s.shape[0] > 1 else self.conv2(s))           # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)) if s.shape[0] > 1 else self.conv3(s))           # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)) if s.shape[0] > 1 else self.conv4(s))           # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s)) if s.shape[0] > 1 else self.fc1(s)), p=0.3, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s)) if s.shape[0] > 1 else self.fc2(s)), p=0.3, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1
        self._value = torch.tanh(v)


        return F.softmax(pi, dim=1), self._value



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
        map_location = 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])

def initializer(tensor, std=0.001):
    
    tensor.data.normal_(0, 1)
    tensor.data *= std / torch.sqrt(
        tensor.data.pow(2).sum(1, keepdim=True))

class MCGardnerNNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        nn.Module.__init__(self)
        
        # game params
        self.game = GardnerMiniChessGame()
        # self.board_x, self.board_y = (self.game.width, self.game.height)
        self.board_x, self.board_y = (5, 5)
        self.action_size = self.game.getActionSize()


        num_channels = 512
        self.num_channels = num_channels

        
        self.conv1 = nn.Conv2d(1, num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, stride=1)

        # self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)

        self.fc1 = nn.Linear(num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        initializer(self.fc1.weight)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        initializer(self.fc2.weight)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        initializer(self.fc3.weight)

        self.fc4 = nn.Linear(512, 1)
        initializer(self.fc4.weight)

        # self.load_checkpoint("~/", "/Users/shiningsunnyday/Desktop/2021-2022/Fall Quarter/AA 228/Final Project/mcts-chess/checkpoint/epoch_1_testloss_153.707695")


    def forward(self, input_dict, state, seq_lens):
        s = input_dict["obs"]["board"].float()     
        indices = input_dict["obs"]["actions"].float()
        test = (s == 0.0).all()
        assert test or not (indices.sum(dim=1) == 0).all()
            # s not all zeros yet indices sum to 0
        # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y) # batch_size x 1 x board_x x board_y
        # s = self.bn0(s) if s.shape[0] > 1 else s
        s = F.relu(self.bn1(self.conv1(s)) if s.shape[0] > 1 else self.conv1(s))           # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)) if s.shape[0] > 1 else self.conv2(s))           # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)) if s.shape[0] > 1 else self.conv3(s))           # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)) if s.shape[0] > 1 else self.conv4(s))           # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s)) if s.shape[0] > 1 else self.fc1(s)), p=0.3, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s)) if s.shape[0] > 1 else self.fc2(s)), p=0.3, training=self.training)  # batch_size x 512
        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1
        self._value = v

        # pi = torch.ones_like(pi)
        pi = F.softmax(pi, dim=1) # batch_size x action_size

        temp = pi.clone().detach()
        pi = pi * indices

        pi = pi / pi.sum(dim=1, keepdim=True)
        
        # print("Here's a pi", "with shape", pi.shape, "max", pi.max(dim=0), "min", pi.min(dim=0))

        # if not test:
        #     print("TEST IS OVER!")
        if test:
            print("S was all 0")
            pi = torch.ones_like(pi)
        elif (indices.sum(dim=1) == 0).all():
            print("indices were all 0...")
        # pi = torch.nan_to_num(pi)

        
        # if torch.argmax(pi) == 0:
        #     print(temp)        
        # if not test:
        #     print("nonzero", torch.nonzero(pi))
        # print("HERE IS YOUR NONZERO PI'S ARGMAX", torch.argmax(pi))
        
        return pi, []

    def value_function(self):
        v = torch.reshape(self._value, [-1])
        # print("Here's a v", "with shape", v.shape, "max", v.max(dim=0), "min", v.min(dim=0))
        return v
        

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
        map_location = 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.load_state_dict(checkpoint['state_dict'])


class ActorCriticModel(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.preprocessor = get_preprocessor(obs_space)(
            obs_space)

        self.shared_layers = None
        self.actor_layers = None
        self.critic_layers = None

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._value_out = self.critic_layers(x)
        return logits, None

    def value_function(self):
        return self._value_out

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value

class DenseModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features=obs_space.shape[0],
                out_features=256), nn.Linear(
                    in_features=256, out_features=256))
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        self._value_out = None

class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        # print("Here's a fc_out", "with shape", fc_out.shape, "max", fc_out.max(dim=0), "min", fc_out.min(dim=0))

        return fc_out, []

    def value_function(self):
        v = torch.reshape(self.torch_sub_model.value_function(), [-1])
        print("Here's a v", "with shape", v.shape, "max", v.max(dim=0), "min", v.min(dim=0))
        return v


class Games(Dataset):
    def __init__(self, path_x, path_pi, path_y):
        self.data = np.load(path_x)
        self.pi = np.load(path_pi)
        self.y = np.load(path_y)

    def __getitem__(self, idx):
        return self.data[idx], self.pi[idx], self.y[idx]

    def __len__(self):
        return len(self.y) 

def train(num_epochs=10,checkpoint=None):
    config = {"num_channels": 512, "dropout": 0.3}
    net = MCGardnerNNetTrain()
    if checkpoint:
        net.load_checkpoint(filename=checkpoint)


    path_x_train = "data/checkpoint_0_train_x.npy"
    path_y_train = "data/checkpoint_0_train_y.npy"
    path_pi_train = "data/checkpoint_0_train_pis.npy"
    path_x_test = "data/checkpoint_0_test_x.npy"
    path_y_test = "data/checkpoint_0_test_y.npy"
    path_pi_test = "data/checkpoint_0_train_pis.npy"

    g_train = Games(path_x_train, path_pi_train, path_y_train)
    g_test = Games(path_x_test, path_pi_test, path_y_test)
    train = DataLoader(g_train, batch_size=100, shuffle=True)
    test = DataLoader(g_test, batch_size=100)

    optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)

    loss_v = nn.MSELoss()

    loss_pi = lambda outputs, targets: -torch.sum(targets * torch.log(outputs)) / targets.size()[0]

    best = float("inf")

    for i in range(num_epochs):
        total_test_loss = 0.0
        net.training = False
        with torch.no_grad():
            for (batch, pis, y) in tqdm(test): 
                batch, pis, y = batch.float(), pis.float(), y.float()
                pi, out = net.forward(batch)

                l_v = loss_v(out, y)
                l_pi = loss_pi(pi, pis)
                loss = l_v + l_pi
                
                total_test_loss += loss.item()

        print("Epoch %d, Test Loss: %f" % (i, total_test_loss))
        if total_test_loss < best:
            best = total_test_loss
            net.save_checkpoint(filename='epoch_%d_testloss_%f' % (i, total_test_loss))


        total_train_loss = 0.0
        net.training = True

        for (batch, pis, y) in tqdm(train):

            batch, pis, y = batch.float(), pis.float(), y.float()
            optimizer.zero_grad()
            pi, out = net.forward(batch)
            l_v = loss_v(out, y)
            l_pi = loss_pi(pi, pis)
            loss = l_v + l_pi
            if loss == loss:
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                print(loss.item())
            else:
                print("NO")

        print("Epoch %d, Train Loss: %f" % (i, total_train_loss))

        

            





if __name__ == "__main__":
    config = {"num_channels": 512, "dropout": 0.3, "cuda": False}
    # checkpoint = "/Users/shiningsunnyday/Desktop/2021-2022/Fall Quarter/AA 228/Final Project/mcts-chess/checkpoint/epoch_1_testloss_151.227986"
    train(num_epochs=10)


    
    