from __future__ import print_function
import sys
from games import Game
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
from games.dark.DarkChessLogic import DarkBoard as Board
import numpy as np
import time
import hashlib

class DarkChessGame(Game):
    RECURSION_LIMIT = 1000
    def __init__(self, n=5):
        self.n = n
        self.setAllActions()

    def getInitBoard(self):
        sys.setrecursionlimit(DarkChessGame.RECURSION_LIMIT)
        # return initial board (numpy board)
        true_b = Board(self.n,
            [
                #  Representation of Gardner Board
                #
                # 5    ♜ ♞ ♝ ♛ ♚
                # 4    ♟ ♟ ♟ ♟ ♟
                # 3    ·  ·  ·  ·  ·
                # 2    ♙ ♙ ♙ ♙ ♙
                # 1    ♖ ♘ ♗ ♕ ♔
                #
                #      a  b  c  d  e

                [-Board.ROOK, -Board.KNIGHT, -Board.BISHOP, -Board.QUEEN, -Board.KING],
                [-Board.PAWN, -Board.PAWN, -Board.PAWN, -Board.PAWN, -Board.PAWN],
                [Board.BLANK, Board.BLANK,   Board.BLANK,   Board.BLANK, Board.BLANK],
                [Board.PAWN, Board.PAWN, Board.PAWN, Board.PAWN, Board.PAWN],
                [Board.ROOK, Board.KNIGHT, Board.BISHOP, Board.QUEEN, Board.KING],
            ]
        )

        obs_b = Board(self.n,
            [
                #  Representation of Gardner Board
                #
                # 5    ♜ ♞ ♝ ♛ ♚
                # 4    ♟ ♟ ♟ ♟ ♟
                # 3    ·  ·  ·  ·  ·
                # 2    ♙ ♙ ♙ ♙ ♙
                # 1    ♖ ♘ ♗ ♕ ♔
                #
                #      a  b  c  d  e

                [Board.BLANK, Board.BLANK,   Board.BLANK,   Board.BLANK, Board.BLANK],
                [Board.BLANK, Board.BLANK,   Board.BLANK,   Board.BLANK, Board.BLANK],
                [Board.BLANK, Board.BLANK,   Board.BLANK,   Board.BLANK, Board.BLANK],
                [Board.PAWN, Board.PAWN, Board.PAWN, Board.PAWN, Board.PAWN],
                [Board.ROOK, Board.KNIGHT, Board.BISHOP, Board.QUEEN, Board.KING],
            ]
        )
        return true_b.pieces_without_padding()

    def setAllActions(self):
        g = GardnerMiniChessGame()
        self.action_to_id = g.action_to_id.copy()
        self.id_to_action = g.id_to_action.copy()

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return len(self.action_to_id)


    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n,board)
        move = self.id_to_action[action]
        b.execute_move(move,player)
        return (b.pieces_without_padding(), -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0.0]*self.getActionSize()
        b = Board(self.n,board)
        if not b.has_legal_moves(player):
            valids[-1]=1.0
            return np.array(valids)
        for (p, x, y) in b.get_legal_moves(player):
            key = str(p)+":"+str(x)+":"+str(y)
            valids[self.action_to_id[key]] = 1.0
        return np.array(valids)

    def getGreedyMove(self,board,player):
        b = Board(self.n, board)
        p,x,y = b.greedy_move(player)
        key = str(p) + ":" + str(x) + ":" + str(y)
        return self.action_to_id[key]

    def getRandomMove(self, board, player):
        b = Board(self.n, board)
        p,x,y = b.random_move(player)
        key = str(p) + ":" + str(x) + ":" + str(y)
        return self.action_to_id[key]

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n,board)
        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves(player):
            return 0
        # draw has a very little value
        return 1e-4

    def getCanonicalForm(self, board, player):
        # add "darkness" to the board (make tiles we cant move to 0)
        value =  [[j*player for j in i] for i in board]
        return value

    def getDarkness(self, board, player):
        # add "darkness" to the board (make tiles we cant move to 0)
        b = Board(self.n, board)
        return b.add_darkness(player)

    def getSymmetries(self, board, pi):
        # mirror, rotational
        # assert(len(pi) == self.n**2+1)  # 1 for pass
        return [(board, pi)]

    def stringRepresentation(self, board):
        return hashlib.md5(np.array_str(np.array(board)).encode('utf-8')).hexdigest()

    def display(self, board, player):
        Board(self.n, board).display(player)


def display(game,board,player):
    Board(game.n,board).display(player)
