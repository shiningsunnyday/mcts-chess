import os
import time
import random

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
from algorithm.multi_env import MultiAgentMinichessEnv

from utils import convert_bit_mask

from games.gardner import GardnerMiniChessGame

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.utils.exploration import *

from mini_train import *

from game.abstract.board import AbstractBoardStatus
from game.board import GardnerChessBoard

from algorithm.env import *

from train import MinichessTrainer

PIECES = {100: "PAWN",
    280: "KNIGHT",
    320: "BISHOP",
    479: "ROOK",
    929: "QUEEN",
    60000: "KING",
    0: " "}

def cell_to_position(x):
    i = (x-15)//7
    j = chr(ord('a') + (x-15-i*7))
    return str(i) + j

def format_legal_moves(env):
    moves = [env.game.id_to_action[x] for x in env.legal_moves]
    return [[PIECES[move[0]], 
    cell_to_position(move[2 if env.player == -1 else 1]), cell_to_position(move[1 if env.player == -1 else 2])] for move in moves]


def run_loop(path):
    env = MultiAgentMinichessEnv(None)
    trainer = MinichessTrainer(None)
    trainer.load(path)
    agent = trainer.agent.get_policy("-1")

    done = {}
    done["__all__"] = False
    while not done["__all__"]:
        print(env.game.display(env.board, env.player))
        # move = input("Please enter move amongst {}:".format(list(zip(range(100), format_legal_moves(env)))))
        # if not len(move): break      
        move = random.choice(list(env.legal_moves))  
        _, _, done, _ = env.step({str(env.player): move})
        if done["__all__"]:
            break

        obs = env._obs()[str(env.player)]
        mov = agent.compute_single_action(obs)[0]
        print(mov)
        _, reward, done, _ = env.step({str(env.player): mov})

    print(env.game.display(env.board, env.player))
    print("done",reward)
        


if __name__ == "__main__":
    run_loop("/Users/gogetaashame/Documents/CS238/mcts-chess/checkpoint/PPO_2021-11-28_13-35-48/PPO_MultiAgentMinichessEnv_01d3f_00000_0_2021-11-28_13-35-49/checkpoint_000002/checkpoint-2")
