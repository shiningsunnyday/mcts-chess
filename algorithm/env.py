# from game.abstract.board import AbstractBoardStatus
# from game.action import GardnerChessAction
# from game.board import GardnerChessBoard
from games.gardner.GardnerMiniChessGame import *

import numpy as np
import gym
from gym.spaces import Discrete, Dict, Box

from games.gardner.GardnerMiniChessLogic import Board

class MinichessEnv(gym.Env):
    def __init__(self, config) -> None:
        self.reset()
        self.action_space = Discrete(self.game.getActionSize())
        self.observation_space = Dict({
            "board": Box(-60000, 60000, shape=(5,5)),
            "actions": Box(low=0, high=1, shape=(self.action_space.n,)),
        })
        

    def reset(self):
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.legal_moves = self._get_legal_actions()
        self.player = 1
        self.steps = 0

        return self._obs()


    def step(self, action):
        if not action in self.legal_moves:
            return self._obs(), -0.01, False, {}
        elif self.steps == 50:
            print("50 steps")
            return self._obs(), -0.001, True, {}

        self.board, self.player = self.game.getNextState(self.board, self.player, action)
        self.legal_moves = self._get_legal_actions()
        obs = self._obs()
        reward = self.game.getGameEnded(self.board, 1)
        done = reward != 0
        if len(self.legal_moves) == 0:
            print(done)
            print(self.game.display(self.board, self.player))
        self.steps += 1

        return obs, reward, done, {}


    def _get_legal_actions(self, return_type="list"):
        legal_moves = self.game.getValidMoves(self.board, self.player, return_type=return_type)
        if return_type != "list":
            print(legal_moves.shape)
        return set(legal_moves) if return_type == "list" else legal_moves
        

    def _obs(self):
        board = np.array(self.board, dtype=np.int64)
        actions = self._get_legal_actions(return_type="one_hot")
        
        assert self.observation_space["actions"].contains(actions)
        assert self.observation_space["board"].contains(board)
        return {
            "board": board,
            "actions": actions,
        }