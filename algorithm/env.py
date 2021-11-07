from game.abstract.board import AbstractBoardStatus
from game.action import GardnerChessAction
from game.board import GardnerChessBoard

import numpy as np
import gym

class MinichessEnv(gym.Env):
    def __init__(self, config) -> None:
        self.action_space = gym.spaces.Discrete(1225)
        self.observation_space = gym.spaces.Box(0, 1, shape=(25 * 12,))
        self.reset()

    def reset(self):
        self.board = GardnerChessBoard()
        self.legal_moves = self._get_legal_actions()
        return self._obs()

    def step(self, action):
        if not action in self.legal_moves:
            return self._obs(), -10, False, {}

        a = GardnerChessAction.decode(int(action), self.board)

        self.board.push(a)
        self.legal_moves = self._get_legal_actions()

        obs = self._obs()
        reward = self.board.reward() / 10000
        done = self.board.status != AbstractBoardStatus.ONGOING

        return obs, reward, done, {}

    def _get_legal_actions(self):
        legal_moves = self.board.legal_actions()
        return {move.idx() for move in legal_moves}
        
    def _obs(self):
        return self.board.state_vector().flatten()