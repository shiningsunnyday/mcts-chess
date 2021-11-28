import os
import time
import random

import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.ddpg as ddpg
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.agents.a3c.a3c import A3CTrainer
from ray.rllib.utils.exploration import *
from algorithm.multi_env import MultiAgentMinichessEnv

from mini_train import *

from game.abstract.board import AbstractBoardStatus
from game.board import GardnerChessBoard

from algorithm.env import *


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

    config["env"] = MultiAgentMinichessEnv
    config["num_gpus"] = 0

    config["framework"] = "torch"
    config["num_workers"] = 1
    config["explore"] = True

    config["exploration_config"] = {"type": "StochasticSampling", "action_space": Discrete(GardnerMiniChessGame().getActionSize()), "random_timesteps": 0, "model": MCGardnerNNet, "framework": "torch"}

    config["train_batch_size"]=1000
    config["sgd_minibatch_size"]=100

    stop = {
        "timesteps_total": 5000000,
    }

    config["model"]["custom_model"] = "gardner_nn"

    config["multiagent"] = {
        "policies": {"-1", "1"},
        "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
    }

    print("Training with Ray Tune")

    results = tune.run("PPO", name="torch_custom", config=config, stop=stop)

    ray.shutdown()