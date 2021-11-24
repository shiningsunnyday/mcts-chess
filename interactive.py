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


def run_loop(env):
    config = ppo.DEFAULT_CONFIG.copy()

    config["env"] = MinichessEnv
    config["num_gpus"] = 0

    config["framework"] = "torch"
    config["num_workers"] = 1
    config["explore"] = True

    config["exploration_config"] = {"type": "StochasticSampling", "action_space": Discrete(GardnerMiniChessGame().getActionSize()), "random_timesteps": 0, "model": MCGardnerNNet, "framework": "torch"}

    config["train_batch_size"]=8
    config["sgd_minibatch_size"]=4

    stop = {
        "timesteps_total": 8,
    }
    ModelCatalog.register_custom_model("gardner_nn", MCGardnerNNet)
    config["model"]["custom_model"] = "gardner_nn"

    agent = ppo.PPOTrainer(config)
    agent.restore("/Users/shiningsunnyday/ray_results/PPO_minichess_2021-11-23_20-08-459otj6ao8/checkpoint_000001/checkpoint-1")
    

    

    done = False
    while not done:

        print(env.game.display(env.board, env.player))
        move = input("Please enter move amongst {}:".format(list(zip(range(100), format_legal_moves(env)))))
        if not len(move): break        
        env.step(list(env.legal_moves)[int(move)])       
        obs=env._obs()
        mov = agent.compute_action(obs)


        _, reward, done, _ = env.step(mov)
    print("done",reward)
        


if __name__ == "__main__":
    env=MinichessEnv({})
    run_loop(env)
