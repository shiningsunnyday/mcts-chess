from game.pieces import King
from game.abstract.board import AbstractChessBoard
from game.abstract.piece import AbstractChessPiece
from game.abstract.board import AbstractChessBoard
from game.abstract.piece import AbstractChessPiece

import numpy as np

from typing import List, Union
from enum import Enum

class AbstractActionFlags(Enum):
    PROMOTE_ROOK = 0
    PROMOTE_KNIGHT = 1
    PROMOTE_BISHOP = 2
    PROMOTE_QUEEN = 3

    CAPTURE = 4
    KING_CAPTURE = 5

    CHECK = 6
    CHECKMATE = 7

    # TODO

class AbstractChessAction:
    '''
        An abstract data type representing a chess action,
        agnostic of rules.

        In Gardner MiniChess, there are 1225 possible actions. We have a 5x5 board from which to choose, and 8 possible
        directions to move any piece at most 4 tiles in that direction. We also have 9 types of underpromotion (3 types
        of move to reach last rank, 3 types of promotion for each move). This provides:
            (5 * 5)( (8 * 4) + 8 + 9) = 1225
        possible moves.
    '''
    def __init__(self, agent: AbstractChessPiece, from_pos: tuple, to_pos: tuple, captured_piece: AbstractChessPiece = None, modifier_flags: List[AbstractActionFlags] = None):
        self.agent = agent
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.captured_piece = captured_piece
        self.modifier_flags = [] if modifier_flags is None else modifier_flags

        if self.captured_piece is not None and type(self.captured_piece) == King:
            self.modifier_flags.append(AbstractActionFlags.KING_CAPTURE)

    def encode(self) -> np.array:
        '''
            Encodes this action as a onehot in a shape (1225,) numpy array.

            Returns
            -------
            A numpy array onehot representing the encoding of this action.
        '''
        raise NotImplementedError

    @staticmethod
    def decode(encoding: Union[np.array, int], state_tm1: np.array):
        '''
            Decode an AbstractChessAction from an actionspace vector and a game state.

            Parameters
            ----------
            encoding :: np.array OR int : the (ACTION_SPACE,) vector representing a given action, or an int representing the index

            state_tm1 :: np.array : the (t-1)-th state of the game, representing the state immediately preceding
            this action. That is, the application of this action drove the game state from `state_tm1` to `state_t`

            Returns
            -------
            AbstractChessAction representing the decoded vector.
        '''
        raise NotImplementedError

    @staticmethod
    def parse(action_string: str, state_tm1: np.array):
        '''
            Parse an AbstractChessAction from a string and a game state.

            Parameters
            ----------
            action_string :: str : the string representation of a given action.

            state_tm1 :: np.array : the (t-1)-th state of the game, representing the state immediately preceding
            this action. That is, the application of this action drove the game state from `state_tm1` to `state_t`

            Returns
            -------
            AbstractChessAction representing the parsed action.
        '''
        raise NotImplementedError

    def fliplr(self):
        '''
            Returns
            -------
            Returns this action rotated about the y-axis. E.g. moving up-right diagonal one is not up-left diagonal one.
        '''
        raise NotImplementedError

    def flipud(self, action):
        '''
            Returns
            -------
            Returns this action rotated about the y-axis. E.g. moving up-right diagonal one is not up-left diagonal one.
        '''
        raise NotImplementedError

    def copy(self):
        '''
            Returns a deep copy of this action.
        '''
        return type(self)(self.agent.copy(), self.from_pos, self.to_pos, self.captured_piece.copy() if self.captured_peice else None, self.modifier_flags.copy() if self.modifier_flags else [])

    def __str__(self):
        return '[ {} -> {} ]'.format(self.from_pos, self.to_pos)

    def __eq__(self, o: object) -> bool:
        return type(o) == type(self) and o.from_pos == self.from_pos and o.to_pos == self.to_pos

    def __hash__(self) -> int:
        return hash((self.from_pos, self.to_pos))

### AbstractActionVisitor for ease of rule-change

# https://stackoverflow.com/a/28398903/7042418

# A couple helper functions first

def _qualname(obj):
    """Get the fully-qualified name of an object (including module)."""
    return obj.__module__ + '.' + obj.__qualname__

def _declaring_class(obj):
    """Get the name of the class that declared an object."""
    name = _qualname(obj)
    return name[:name.rfind('.')]

# Stores the actual visitor methods
_methods = {}

# Delegating visitor implementation
def _visitor_impl(self, arg, arg2):
    """Actual visitor method implementation."""
    method = _methods[(_qualname(type(self)), type(arg))]
    return method(self, arg, arg2)

# The actual @visitor decorator
def visitor(arg_type):
    """Decorator that creates a visitor method."""

    def decorator(fn):
        declaring_class = _declaring_class(fn)
        _methods[(declaring_class, arg_type)] = fn

        # Replace all decorated methods with _visitor_impl
        return _visitor_impl

    return decorator

class AbstractChessActionVisitor:
    '''
        Skeleton code for our ChessActionVisitor, a class designed for ease of use in modifying and changing rules.

        For changing game rules, we need simply implement a new subclass of AbstractChessActionVisitor.
    '''
    @visitor(AbstractChessPiece)
    def visit(self, piece: AbstractChessPiece, board: AbstractChessBoard) -> List[AbstractChessAction]:
        '''
            Generates a list of legal actions for this piece given the current board.

            This must be implemented for every class of piece for any given chess variant. See
            minichess.gardner.action for a with-code example of final state.

            Parameters
            ----------
            piece :: AbstractChessPiece : the piece to check legal moves for

            board :: AbstractChessBoard : the current board to check for moves on

            Returns
            -------
            A list of AbstractChessActions representing all possible legal moves for this piece.
        '''
        raise NotImplementedError

