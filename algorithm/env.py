# from game.abstract.board import AbstractBoardStatus
# from game.action import GardnerChessAction
# from game.board import GardnerChessBoard
import random
from games.gardner.GardnerMiniChessGame import *

import numpy as np
import gym
from gym.spaces import Discrete, Dict, Box
import pdb

from games.gardner.GardnerMiniChessLogic import Board

class MinichessEnv(gym.Env):
    def __init__(self, config) -> None:
        self.game = GardnerMiniChessGame()
        self.board = self.game.getInitBoard()
        self.player = 1
        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        
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

        return self._obs()


    def step(self, action):
        if not action in self.legal_moves:
            print("action", action,"not in", self.legal_moves)
            # print([self.game.id_to_action[move] for move in self.legal_moves])
            print(self.game.display(self.board,self.player))
            print(self.player)
            print(self.game.id_to_action[action], "BAD ACTION")
            assert False
            return self._obs(), -0.01, False, {}
        elif self.steps == 100:
            # print("50 steps")
            return self._obs(), -0.1, True, {}

        self.board, self.player = self.game.getNextState(self.board, self.player, action)
        reward = self.game.getGameEnded(self.board, 1)
        done = reward != 0
        
        # if done:
        #     print(self.steps)

        if not done:
            # Play random move for other agent
            legal_moves = list(self._get_legal_actions())
            move = random.choice(legal_moves)
            self.board, self.player = self.game.getNextState(self.board, self.player, move)
            reward = self.game.getGameEnded(self.board, 1)
            done = reward != 0

        self.legal_moves = self._get_legal_actions()
        self.legal_moves_one_hot = self._get_legal_actions(return_type="one_hot")
        obs = self._obs()

        reward = np.sum(obs["board"]) / 1000

        
        if done:
            # print(self.game.display(self.board, self.player))
            # print("\nGAME OVER {}\n".format(reward)) 
            pass

        self.steps += 1

        return obs, reward, done, {}


    def _get_legal_actions(self, return_type="list"):
        legal_moves = self.game.getValidMoves(self.board, self.player, return_type=return_type)        # if return_type != "list":
        #     print(legal_moves.shape)
        return set(legal_moves) if return_type == "list" else legal_moves
        

    def _obs(self):
        board = np.array(self.board, dtype=np.int32)

        assert not (board == 0).all()
        actions = np.array(self.legal_moves_one_hot,dtype=np.int32)
        if not self.observation_space["board"].contains(board):
            pdb.set_trace()
        
        assert self.observation_space["actions"].contains(actions)
        
        assert self.observation_space["board"].contains(board)
        # print("feeding legal moves", [self.game.id_to_action[move] for move in self.legal_moves])
        return {
            "board": board,
            "actions": actions
        }