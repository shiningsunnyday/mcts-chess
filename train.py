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

ModelCatalog.register_custom_model("gardner_nn", MCGardnerNNet)

ENV = MinichessEnv

weights = {}

def run_loop(n, path):
    env = ENV(None)
    trainer = MinichessTrainer()
    trainer.load(path)
    trainer.agent.set_weights(weights)
    agent = trainer.agent.get_policy("1")

    wins = 0
    for i in range(n):
        env.reset()
        done = {}
        done["__all__"] = False
        while not done["__all__"]:
            # move = input("Please enter move amongst {}:".format(list(zip(range(100), format_legal_moves(env)))))
            # if not len(move): break      
            obs = env._obs()[str(env.player)]
            mov = agent.compute_single_action(obs)[0]
            _, reward, done, _ = env.step({str(env.player): mov})
            if done["__all__"]:
                break

            move = random.choice(list(env.legal_moves))  
            _, _, done, _ = env.step({str(env.player): move})

        print(env.game.display(env.board, env.player))
        if env.game.getGameEnded(env.board, 1) == 1:
            wins += 1
    print("done",reward)

    trainer.agent.export_policy_model("export", "-1")
    trainer.agent.export_policy_model("export", "1")

    return wins / n


class MinichessGenerativeTrainer:
    def __init__(self, num_iter) -> None:
        self.num_iter = num_iter
        self.wr_lst = []
    
    def run(self, stop):
        config = None
        fixed_player = -1
        for i in range(self.num_iter):
            print(f"Training Iteration {i}")
            new_weights = {}
            for player in ["1", "-1"]:
                print(f"Training player {player}")
                trainer = MinichessTrainer(config, fixed_player)
                path, analysis = trainer.train(stop)

                new_weights[player] = trainer.agent.get_weights()[player]
                trainer.agent.export_policy_model("export", player)

                fixed_player = -fixed_player
                if player == "1":
                    self.wr_lst.append(run_loop(1000, path))
                    print(self.wr_lst)
            weights["1"] = new_weights["1"]
            weights["-1"] = new_weights["-1"]             

        print(self.wr_lst)   
        # return weights


class MinichessTrainerWrapper(ppo.PPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_weights(weights)


class MinichessTrainer:
    def __init__(self, config = None, fixed_player = -1) -> None:
        self.env_class = ENV
        if config is None:
            config = ppo.DEFAULT_CONFIG.copy()

            config["env"] = ENV
            config["num_gpus"] = 1

            config["framework"] = "torch"
            config["num_workers"] = 10
            config["explore"] = True

            config["exploration_config"] = {"type": "StochasticSampling", "action_space": Discrete(GardnerMiniChessGame().getActionSize()), "random_timesteps": 0, "model": MCGardnerNNet, "framework": "torch"}
            config["lr"] = 1e-5
            config["train_batch_size"] = 1000
            config["sgd_minibatch_size"] = 100


            config["model"]["custom_model"] = "gardner_nn"
            config["model"]["custom_model_config"] = {"checkpoint": ""}

            # config["multiagent"] = {
            #     "policies": {
            #         "-1": self._gen_policy(0 if fixed_player == -1 else 1e-3),
            #         "1": self._gen_policy(0 if fixed_player == 1 else 1e-3),
            #     },
            #     "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
            # }

        self.config = config
        self.save_dir = "checkpoint"
        self.env_config = None

        # self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
        # if weights is not None and len(weights) == 2:
        #     self.agent.set_weights(weights)

    def _gen_policy(self, lr=1e-5):
        config = {
            "model": {
                "custom_model": "gardner_nn",
            },
            "lr": lr,
        }
        env = self.env_class(None)
        return (None, env.observation_space, env.action_space, config)

    def train(self, stop_criteria):
        """
        Train an RLlib PPO agent using tune until any of the configured stopping criteria is met.
        :param stop_criteria: Dict with stopping criteria.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        # results = self.agent.train()
        # self.agent.log_result(results)
        # return None, results
        analysis = tune.run(MinichessTrainerWrapper, name="minichesstrainerwrapper", config=self.config, local_dir=self.save_dir, stop=stop_criteria,
                                checkpoint_at_end=True)
        # list of lists: one list per checkpoint; each checkpoint list contains 1st the path, 2nd the metric value
        checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean', mode="max"),
                                                        metric='episode_reward_mean')
        # retriev the checkpoint path; we only have a single checkpoint, so take the first one
        checkpoint_path = checkpoints[0][0]
        self.load(checkpoint_path)
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
    # config = ppo.DEFAULT_CONFIG.copy()

    # config["env"] = ENV
    # config["num_gpus"] = 0

    # config["framework"] = "torch"
    # config["num_workers"] = 1
    # config["explore"] = True

    # config["exploration_config"] = {"type": "StochasticSampling", "action_space": Discrete(GardnerMiniChessGame().getActionSize()), "random_timesteps": 0, "model": MCGardnerNNet, "framework": "torch"}

    # config["train_batch_size"]= 64
    # config["sgd_minibatch_size"]= 8
    
    stop = {
        "timesteps_total": 50000,
    }

    # config["model"]["custom_model"] = "gardner_nn"
    # config["model"]["custom_model_config"] = {"checkpoint": args.critic_checkpoint}

    # config["multiagent"] = {
    #     "policies": {"-1", "1"},
    #     "policy_mapping_fn": lambda agent_id, episode, **kwargs: agent_id,
    # }

    print("Training with Ray Tune")

    # results = tune.run("PPO", name="torch_custom", config=config, stop=stop)

    # ray.shutdown()
    runner = MinichessGenerativeTrainer(50)

    # trainer = MinichessTrainer(config)
    # path, analysis = trainer.train(stop)
    # print(path)

    # trainer.load(path)
    # print(trainer.agent.get_policy("1"))
    # print(trainer.agent.get_policy("-1"))
    runner.run(stop)

    
