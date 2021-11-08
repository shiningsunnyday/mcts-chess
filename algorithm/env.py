# from game.abstract.board import AbstractBoardStatus
# from game.action import GardnerChessAction
# from game.board import GardnerChessBoard
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

import numpy as np
import gym

from games.gardner.GardnerMiniChessLogic import Board

class MinichessEnv(gym.Env):
    def __init__(self, config) -> None:
        self.reset()
        self.action_space = gym.spaces.Discrete(self.game.getActionSize())
        self.observation_space = gym.spaces.Box(-60000, 60000, shape=(5,5))

    def reset(self):
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.legal_moves = self._get_legal_actions()
        self.player = 1
        return self._obs()

    def step(self, action):
        if not action in self.legal_moves:
            return self._obs(), -10, False, {}

        self.board, self.player = self.game.getNextState(self.board, self.player, action)
        self.legal_moves = self._get_legal_actions()

        obs = self._obs()
        reward = self.game.getGameEnded(self.board, self.player) * 10000
        done = reward != 0

        return obs, reward, done, {}

    def _get_legal_actions(self):
        legal_moves = self.game.getValidMoves(self.board, 1, return_type="list")
        return set(legal_moves)
        
    def _obs(self):
        return self.board