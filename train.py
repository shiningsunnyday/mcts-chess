import os
import time
import random

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
from ray.tune.registry import register_env
def env_creator(env_config):
    return MinichessEnv(env_config)

register_env("minichess", env_creator)

def ppo_train(config):
    agent = ppo.PPOTrainer(env="minichess", config=config)
    checkpoint_path = ""
    while True:
        result = agent.train()

        checkpoint_path = agent.save()
        print("saved at",checkpoint_path)
    
        break 

    return checkpoint_path

        #you can also change the curriculum here

if __name__ == "__main__":
    # g = GardnerChessBoard()
    
    # while g.status == AbstractBoardStatus.ONGOING:
    #     print(g)
    #     print(g.state_vector())

    #     actions = g.legal_actions()

    #     g.push(random.choice(actions))
    #     print('+---------------+')
    #     time.sleep(0.1)

    # print(g)
    # print(g.status)
    # print(g.state_vector())

    ray.init(ignore_reinit_error=False)

    ModelCatalog.register_custom_model("gardner_nn", MCGardnerNNet)
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

    config["model"]["custom_model"] = "gardner_nn"

    print("Training with Ray Tune")

    path = ppo_train(config)
    
    
    
    
    ray.shutdown()