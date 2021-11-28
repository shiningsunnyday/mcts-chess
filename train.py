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
import argparse

class MinichessTrainer:
    def __init__(self, config) -> None:
        self.config = config
        self.save_dir = "checkpoint"
        self.env_class = MultiAgentMinichessEnv
        self.env_config = None

    def train(self, stop_criteria):
        """
        Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
        :param stop_criteria: Dict with stopping criteria.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        analysis = ray.tune.run(ppo.PPOTrainer, config=self.config, local_dir=self.save_dir, stop=stop_criteria,
                                checkpoint_at_end=True)
        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean', mode="max"),
                                                        metric='episode_reward_mean')
        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        return checkpoint_path, analysis

    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
        self.agent.restore(path)

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        env = self.env_class(self.env_config)

        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = self.agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        return episode_reward


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--critic_checkpoint',type=str,help="path to checkpoint")
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

    # ray.init(ignore_reinit_error=False)

    ModelCatalog.register_custom_model("gardner_nn", MCGardnerNNet)
    config = ppo.DEFAULT_CONFIG.copy()

    config["env"] = MultiAgentMinichessEnv
    config["num_gpus"] = 1

    config["framework"] = "torch"
    config["num_workers"] = 10
    config["explore"] = True

    config["exploration_config"] = {"type": "StochasticSampling", "action_space": Discrete(GardnerMiniChessGame().getActionSize()), "random_timesteps": 0, "model": MCGardnerNNet, "framework": "torch"}

    config["train_batch_size"]=32
    config["sgd_minibatch_size"]=4
    config["entropy_coeff"]=0.00
    config["lr"] = 1e-5
    
    stop = {
        "timesteps_total": 64,
    }

    config["model"]["custom_model"] = "gardner_nn"
    config["model"]["custom_model_config"] = {"checkpoint": args.critic_checkpoint}

    config["multiagent"] = {
        "policies": {"-1", "1"},
        "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
    }

    print("Training with Ray Tune")

    # results = tune.run("PPO", name="torch_custom", config=config, stop=stop)

    # ray.shutdown()

    trainer = MinichessTrainer(config)
    path, analysis = trainer.train(stop)

    trainer.load(path)
    print(trainer.agent.get_policy("1"))
    print(trainer.agent.get_policy("-1"))
