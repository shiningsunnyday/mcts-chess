# from game.abstract.board import AbstractBoardStatus
# from game.action import GardnerChessAction
# from game.board import GardnerChessBoard
import random
from games.gardner.GardnerMiniChessGame import *

import numpy as np
from gym.spaces import Discrete, Dict, Box
import pdb

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class MultiAgentMinichessEnv(MultiAgentEnv):
    def __init__(self, config) -> None:
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.player = 1
        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        self.agents = [-1, 1]

        self.steps = 0
        self.action_space = Discrete(self.game.getActionSize())
        self.observation_space = Dict({
            "board": Box(-60000, 60000, shape=(5,5), dtype=np.int32),
            "actions": Box(low=0, high=1, shape=(self.action_space.n,), dtype=np.int32),
        })
        

    def reset(self):
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.player = 1
        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        
        self.steps = 0

        return self.get_agent_obs()


    def step(self, action_dict):
        action = action_dict[str(self.player)]

        if not action in self.legal_moves:
            print("action", action,"not in", self.legal_moves)
            # print([self.game.id_to_action[move] for move in self.legal_moves])
            print(self.game.display(self.board,self.player))
            print(self.player)
            print(self.game.id_to_action[action], "BAD ACTION")
            assert False
        elif self.steps == 100:
            obs = self._obs()
            rewards = {
                str(self.player): 0,
                str(-self.player): 0,
            }
            dones = {
                str(self.player): True,
                str(-self.player): True,
                "__all__": True,
            }
            return obs, rewards, dones, {}

        self.board, self.player = self.game.getNextState(self.board, self.player, action)
        reward = self.game.getGameEnded(self.board, self.player)
        rewards = {
            str(self.player): reward,
            str(-self.player): -reward,
        }

        done = reward != 0
        dones = {
            str(self.player): done,
            str(-self.player): done,
            "__all__": done,
        }
        if done:
            print(self.game.display(self.board, self.player))
            print(self.steps)

        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        obs = self._obs()
        self.steps += 1
        return obs, rewards, dones, {}


    def _get_legal_actions(self, return_type="list"):
        legal_moves = self.game.getValidMoves(self.board, self.player, return_type=return_type)        # if return_type != "list":
        #     print(legal_moves.shape)
        return set(legal_moves) if return_type == "list" else legal_moves

    def get_agent_obs(self):
        return self._obs()

    def _obs(self):
        board = np.array(self.board, dtype=np.int32)

        assert not (board == 0).all()
        actions = np.array(self.legal_moves_one_hot,dtype=np.int32)
        if not self.observation_space["board"].contains(board):
            pdb.set_trace()
        
        assert self.observation_space["actions"].contains(actions)        
        assert self.observation_space["board"].contains(board)

        return {
            str(self.player): {
            "board": board,
            "actions": actions
            }
        }