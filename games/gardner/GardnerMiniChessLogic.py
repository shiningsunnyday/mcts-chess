"""
Board class for the game of MiniChess.
Default board size is 5x5.

Author: Karthik selvakumar, github.com/karthikselva
Date: May 15, 2018.

"""

from __future__ import print_function
import random
import re, sys, time
from itertools import count
from collections import OrderedDict, namedtuple
import numpy as np
from enum import Enum


# To make the piece colors more evident in terminals
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Board:

    # Assigning Initial weights for all the Pieces
    PAWN = 100
    KNIGHT = 280
    BISHOP = 320
    ROOK = 479
    QUEEN = 929
    KING = 60000
    BLANK = 0
    INF = 1000000

    PLAYER1 = 1
    PLAYER2 = -1

    def __init__(self, n, pieces):
        "Set up initial board configuration."
        self.n = n
        self.last_cell = 62
        self.bottom_left = 43
        self.bottom_right = 47
        self.top_left = 15
        self.top_right = 19
        self.pieces = self.add_padding(pieces)
        self.is_rotated = False
        self.player_won = 0
        self.wc = (True, True)
        self.bc = (True, True)
        self.ep = 0
        self.kp = 0

        # Chess GRID with Padding and Cell Number
        # [0,  1,  2,  3,  4,  5,  6]
        # [7,  8,  9,  10, 11, 12, 13]

        # [14,   15, 16, 17, 18, 19,     20]
        # [21,   22, 23, 24, 25, 26,     27]
        # [28,   29, 30, 31, 32, 33,     34]
        # [35,   36, 37, 38, 39, 40,     41]
        # [42,   43, 44, 45, 46, 47,     48]

        # [49, 50, 51, 52, 53, 54, 55]
        # [56, 57, 58, 59, 60, 61, 62]

        self.north, self.east, self.south, self.west = -(self.n+2), 1, (self.n+2), -1
        self.directions = {
            Board.PAWN: (-7, -14, -8, -6),
            Board.KNIGHT: (-9, -15, -13, -5, 15, 13, 9, 5),
            Board.BISHOP: (8, 6, -8, -6),
            Board.ROOK: (-7, 1, 7, -1),
            Board.QUEEN: (8, 6, -8, -6, -7, 1, 7, -1),
            Board.KING: (7, -7, 6, 8, -6, -8, -1, 1)
        }

    

    def add_padding(self, board):
        padded_board = []
        padded_board.append([Board.INF]*(self.n+2))
        padded_board.append([Board.INF]*(self.n+2))
        for row in board:
            padded_board.append([Board.INF] + row + [Board.INF])
        padded_board.append([Board.INF]*(self.n+2))
        padded_board.append([Board.INF]*(self.n+2))
        return padded_board

    def set(self,row,col,piece):
        row = row + 2
        col = col + 1
        self.pieces[row][col] = piece

    def get_legal_moves(self,player):
        moves = []
        attack_moves = []
        flat_pieces = [item for sublist in self.pieces for item in sublist]
        for (p, start, end) in self._get_legal_moves(player):
            moves.append((p,start,end))
            if flat_pieces[end] < 0: attack_moves.append((p,start,end))


        # Reducing the recursion space of MCTS by giving priority to attack moves
        # over passive moves if they do exist otherwise do a passive move
        # if len(attack_moves) > 0:
        #     moves = attack_moves
        return moves

    def greedy_move(self,player):
        moves = []
        attack_move = None
        min_val = 0
        flat_pieces = [item for sublist in self.pieces for item in sublist]
        for (p, start, end) in self._get_legal_moves(player):
            moves.append((p,start,end))
            if flat_pieces[end] < min_val:
                attack_move = (p,start,end)
                min_val = flat_pieces[end]
        if attack_move == None: attack_move = moves[0] # Some random move if no attack move exists
        return attack_move

    def random_move(self, player):
        moves = [x for x in self._get_legal_moves(player)]
        return random.choice(moves)

    def _get_legal_moves(self,player):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        flat_pieces = [item for sublist in self.pieces for item in sublist]
        for i, p in enumerate(flat_pieces):
            if p * player <= 0 or abs(p) == Board.INF: continue
            p = p * player
            for d in self.directions[p]:
                for j in count(i+d, d):
                    q = flat_pieces[j]
                    # Stay inside the board, and off friendly pieces

                    
                    
                    if (q > 0 if player == 1 else q < 0) or abs(q) == Board.INF: break
                    # Pawn move, double move and capture
                    if p == Board.PAWN and d in (self.north, self.north+self.north) and q != Board.BLANK: break
                    if p == Board.PAWN and d == self.north+self.north and (i < self.bottom_left+self.north or flat_pieces[i+self.north] != Board.BLANK): break
                    if p == Board.PAWN and d in (self.north+self.west, self.north+self.east) and q == Board.BLANK and j not in (self.ep, self.kp): break
                    # Move it
                    yield (p, i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in [Board.PAWN,Board.KNIGHT,Board.KING] or (q < 0 if player == 1 else q > 0): break

    def rotate(self,board):
        self.is_rotated = not self.is_rotated
        self.pieces = np.array(board).reshape((self.n+4,self.n+2))
        self.pieces = self.pieces[::-1]*(-1)
        ''' Rotates the board, preserving enpassant '''
        self.ep = self.last_cell-self.ep if self.ep else 0
        self.kp = self.last_cell-self.kp if self.kp else 0
        return (
            self.pieces,self.bc, self.wc,
            self.ep,self.kp)


    def execute_move(self, move, player):
        piece, i, j = move
        i = i
        flat_pieces = [item for sublist in self.pieces for item in sublist]
        p, q = flat_pieces[i], flat_pieces[j]
        put = lambda board, i, p: np.append(np.append(board[:i],[p]),board[i+1:])
        # Copy variables and reset ep and kp
        board = flat_pieces
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        # Actual move
        if abs(flat_pieces[j]) == Board.KING:
            self.player_won = player
        board = put(board, j, board[i])
        board = put(board, i, Board.BLANK)
        # Castling rights, we move the rook or capture the opponent's
        if i == self.bottom_left: wc = (False, wc[1])
        if i == self.bottom_right: wc = (wc[0], False)
        if j == self.top_left: bc = (bc[0], False)
        if j == self.top_right: bc = (False, bc[1])
        # Castling
        if p == Board.KING:
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, self.bottom_left if j < i else self.bottom_right, Board.BLANK)
                board = put(board, kp, Board.ROOK)
        # Pawn promotion, double move and en passant capture
        if p == Board.PAWN:
            if self.top_left <= j <= self.top_right:
                board = put(board, j, Board.QUEEN)
            if j - i == 2*self.north:
                ep = i + self.north
            if j - i in (self.north+self.west, self.north+self.east) and q == Board.BLANK:
                board = put(board, j+self.south, Board.BLANK)
        # We rotate the returned position, so it's ready for the next player

        return self.rotate(board)


    def has_legal_moves(self,player):
        for move in self.get_legal_moves(player):
            return True
        return False

    def is_win(self, player):
        flat_pieces = [item for sublist in self.pieces for item in sublist]
        if player == Board.PLAYER1 and (Board.PLAYER2*Board.KING not in flat_pieces):
            return True
        if player == Board.PLAYER2 and (Board.PLAYER1*Board.KING not in flat_pieces):
            return True
        return False

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def pieces_without_padding(self):
        result = []
        for i, row in enumerate(self.pieces):
            clean_row = list(filter(lambda a: abs(a) != Board.INF, row))
            if len(clean_row) > 0:
                if self.is_rotated:
                    clean_row = [-x for x in clean_row]
                result.append(clean_row)
        return result

    def display(self,player):
        print()
        SPACE = ' '
        uni_pieces = {
            -Board.ROOK: '♜',
            -Board.KNIGHT: '♞',
            -Board.BISHOP: '♝',
            -Board.QUEEN: '♛',
            -Board.KING: '♚',
            -Board.PAWN: '♟',
            Board.ROOK: '♖',
            Board.KNIGHT: '♘',
            Board.BISHOP: '♗',
            Board.QUEEN: '♕',
            Board.KING: '♔',
            Board.PAWN: '♙',
            Board.BLANK: '⊙'
        }
        result = self.pieces_without_padding()
        if player < 0: result = result[::-1]
        color_output = False
        s = ''

        for row in result:
            s += SPACE.join([uni_pieces[tile] for tile in row])
            s += '\n'
        return s
        # print(' ', self.n - i, ' '.join(uni_pieces.get(p, p) for p in row))
        for i, row in enumerate(result):
            out = ''
            if not color_output:
                out = SPACE.join(uni_pieces.get(p*player) for p in row) + '                            '
            else:
                out = []
                for p in row:
                    p = p*player
                    if p > 0:
                        out.append(bcolors.FAIL + uni_pieces.get(p) + bcolors.ENDC)
                    elif p < 0:
                        out.append(bcolors.OKGREEN + uni_pieces.get(p) + bcolors.ENDC)
                    else:
                        out.append(uni_pieces.get(p))
                out = ' '.join(out)
            print(out)
        for i in range(6): sys.stdout.write("\033[F")  # Cursor up one line
        # time.sleep(1)
        # print('    a  b  c  d  e  \n\n')