import os
import time
import random

import ray
from ray import tune
from ray.rllib.agents.a3c.a3c import A3CTrainer

from game.abstract.board import AbstractBoardStatus
from game.board import GardnerChessBoard

from algorithm.env import MinichessEnv


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

    ray.init()
    config = {
        "env": MinichessEnv,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 10,  # parallelism
    }

    stop = {
        "training_iteration": 50,
        "timesteps_total": 100000,
        "episode_reward_mean": 0.1,
    }

    print("Training with Ray Tune")
    results = tune.run("A3C", config=config, stop=stop)

    
    ray.shutdown()