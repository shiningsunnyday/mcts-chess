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
import argparse

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--critic_checkpoint',type=str,help="path to checkpoint")
    parser.add_argument('--exp_name',type=str,help="name of experiment for tensorboard")
    args=parser.parse_args()
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
    config["num_gpus"] = 1

    config["framework"] = "torch"
    config["num_workers"] = 10
    config["explore"] = True

    config["exploration_config"] = {"type": "StochasticSampling", "action_space": Discrete(GardnerMiniChessGame().getActionSize()), "random_timesteps": 0, "model": MCGardnerNNet, "framework": "torch"}

    config["train_batch_size"]=1000
    config["sgd_minibatch_size"]=100
    config["entropy_coeff"]=0.0
    config["gamma"]=0.5
    config["lr"] = 1e-5

    stop = {
        "timesteps_total": 100000,
        "episode_reward_mean": 10000.0
    }

    config["model"]["custom_model"] = "gardner_nn"
    config["model"]["custom_model_config"] = {"checkpoint": args.critic_checkpoint}

    print("Training with Ray Tune")

    analysis = tune.run("PPO", name=args.exp_name, config=config, stop=stop, checkpoint_at_end=True)
    checkpoints = analysis.get_trial_checkpoints_paths(
    trial=analysis.get_best_trial("episode_reward_mean", mode="max"),
    metric="episode_reward_mean"
)
    checkpoint_path = checkpoints[0][0]
    print(checkpoint_path)    
    ray.shutdown()