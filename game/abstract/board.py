import enum
from game.abstract.piece import AbstractChessPiece, PieceColor
from game.abstract.resources import *

import numpy as np

import hashlib

from typing import List, Union, Iterable

LETTER_TO_COLUMN = {
    'e': 0,
    'd': 1,
    'c': 2,
    'b': 3,
    'a': 4
}

class AbstractBoardStatus(enum.Enum):
    ONGOING = 0
    WHITE_WIN = 1
    BLACK_WIN = 2
    DRAW = 3

class AbstractChessBoard:
    '''
        An abstract data type representing a Chess Board,
        agnostic of rules.
    '''
    def __init__(self, side_length):
        self.side_length = side_length
        self.move_history = []
        self._board = self._init_board(side_length, side_length)
        self.active_color = PieceColor.WHITE

    def _init_board(self, height, width):
        self.height = height
        self.width = width

        board = list()

        tile_count = 0
        for row_num in range(height):
            board.append([])
            for col_num in range(width):
                # set our color
                color = PieceColor.BLACK if tile_count % 2 == 0 else PieceColor.WHITE
                
                # add a tile
                board[row_num].append(AbstractChessTile(color, (row_num, col_num), None))

                tile_count += 1

        return board

    @staticmethod
    def from_vector(vector):
        '''
            Decodes a Chess Board from a vector.

            Returns
            -------
            AbstractChessBoard that the vector represents.
        '''
        raise NotImplementedError

    def push(self, action):
        '''
            Push a `AbstractChessAction` to this chess board,
            applying it to the current board and updating
            current state appropriately.

            Parameters
            ----------
            action :: AbstractChessAction : an action to apply to
            this game board.

            Returns
            -------
            AbstractChessBoard reflecting the new state of the board after this action is
            applied.
        '''
        raise NotImplementedError

    def pop(self):
        '''
            Removes the most recent move from this board, undoing
            its effects and returning the move itself.

            Returns
            -------
            A tuple (AbstractChessBoard, AbstractChessAction) representing this board after undoing the most recent action, and the most recent action, respectively.
        '''
        raise NotImplementedError

    def peek(self):
        '''
            Returns
            -------
            The most recent move from this board.
        '''
        raise NotImplementedError

    def get(self, index: Union[str, Iterable[int]]):
        '''
            Returns the tile located at `index`

            Parameters
            ----------
            index :: Iterable[Int, Int] or str : the index to retrieve. A str uses chess convention, an iterable uses python convention.

            Our board looks like this under chess convention:
            ```
                   E   D   C   B   A
                1 [ ] [ ] [ ] [ ] [ ] 1
                2 [ ] [ ] [ ] [ ] [ ] 2
                3 [ ] [ ] [ ] [ ] [ ] 3
                4 [ ] [ ] [ ] [ ] [ ] 4
                5 [ ] [ ] [ ] [ ] [ ] 5
                   E   D   C   B   A
            ```
            and like this under python convention:
            ```
                   0   1   2   3   4
                0 [ ] [ ] [ ] [ ] [ ] 0
                1 [ ] [ ] [ ] [ ] [ ] 1
                2 [ ] [ ] [ ] [ ] [ ] 2
                3 [ ] [ ] [ ] [ ] [ ] 3
                4 [ ] [ ] [ ] [ ] [ ] 4
                   0   1   2   3   4
            ```

            Returns
            -------
            The tile located at `index`.
        '''
        if type(index) == str: # chess convention
            # validate our input
            assert len(index) == 2 and str.isalpha(index[0]) and str.isdigit(index[1])

            alp,num = index[0].lower(), int(index[1])

            row = num - 1
            col = LETTER_TO_COLUMN[alp]
        else:
            # validate our input
            assert len(index) == 2

            row,col = index

        return self._board[row][col]

    def reward(self) -> float:
        '''
            Returns
            -------
            The reward value of the current state of the board.
        '''
        raise NotImplementedError

    def legal_actions(self) -> List:
        '''
            Returns
            -------
            List of AbstractChessActions corresponding to all possible legal moves in current game state.
        '''
        return self.legal_actions_for_color(self.active_color)

    def legal_actions_for_color(self, color: PieceColor) -> List:
        '''
            Parameters
            ----------
            color :: PieceColor : the color for which to check legal moves

            Returns
            -------
            List of AbstractChessActions corresponding to all possible legal moves by color `color` in current game state.
        '''
        raise NotImplementedError

    def legal_action_mask(self) -> np.array:
        '''
            Returns
            -------
            shape (NUM_ACTIONS,) numpy array of 0s and 1s, where a 0 corresponds 
            to an illegal move, and 1 corresponds to a legal move
        '''
        raise NotImplementedError

    def state_vector(self) -> np.array:
        '''
            Outputs a vector representation of this board for use in providing
            some observation to an RL model.

            Returns
            -------
            A numpy array representing the current state of this board.
        '''
        raise NotImplementedError

    def canonical_state_vector(self) -> np.array:
        '''
            Returns the vector representation of this board from the perspective of
            the active color AS IF that active color was white.

            e.g. if active color is white, return normal. Else, invert and flip the board.
        '''
        raise NotImplementedError

    def is_valid_position(self, position: tuple) -> bool:
        '''
            Returns true if this position lies on the game board.
        '''

        row,col = position

        return row in range(0, self.height) and col in range(0, self.width)

    @property
    def status(self) -> AbstractBoardStatus:
        '''
            Returns the `AbstractBoardStatus` of this board.
        '''
        raise NotImplementedError

    def copy(self):
        '''
            Returns
            -------
            A deep copy of this board.
        '''
        raise NotImplementedError

    def __str__(self) -> str:
        s = ''

        for row in self._board:
            s += SPACE.join([str(tile) for tile in row])
            s += '\n'

        return s

    def canonical_str(self) -> str:
        if self.active_color == PieceColor.WHITE:
            return str(self)
        else:
            temp_board = np.fliplr(np.flipud(self._board))

            s = ''

            for row in temp_board:
                row_str = []
                for col in row:
                    piece = col.peek()
                    if piece != None:
                        inv_piece = type(piece)(piece.color.invert(), (-1, -1), 0)
                        row_str.append(str(inv_piece))
                    else:
                        row_str.append(str(col))
                s += SPACE.join(row_str)
                s += '\n'

            return s

    def __iter__(self):
        for row_num in range(self.height):
            for col_num in range(self.width):
                yield self.get((row_num, col_num))

    def __hash__(self):
        return int(hashlib.sha512(str(self).encode('utf-8')).hexdigest(), 16)

class AbstractChessTile:
    '''
        An abstract data type representing a chess tile.

        It has the following properties:

        color :: PieceColor : the color of the tile

        position :: tuple(int, int) : the raw formatted position of this tile, (0,0) is top left corner

        piece :: 
    '''

    def __init__(self, color: PieceColor, position: tuple, piece: AbstractChessPiece) -> None:
        self.color = color
        self.position = position
        self.piece = piece

    def occupied(self) -> bool:
        '''
            Returns
            -------
            True if there is a piece on this tile, False otherwise.
        '''
        return self.piece is not None

    def capturable(self, color: PieceColor) -> bool:
        '''
            Returns
            -------
            True if there is a piece of color != `color` on this tile, False otherwise.
        '''
        return self.piece is not None and self.piece.color != color

    def push(self, piece: AbstractChessPiece):
        '''
            Push a new piece to this tile, removing old occupying pieces.

            Parameters
            ----------
            piece :: AbstractChessPiece : the piece to occupy this tile.
        '''
        self.pop() # clear former piece, if applicable

        self.piece = piece
        if self.piece != None: self.piece.set_position(self.position)

    def pop(self):
        '''
            Removes (and returns) the piece from this tile.
        '''
        
        if self.occupied():
            piece = self.piece
            self.piece = None

            # COMMENTED OUT BECAUSE state is important to maintain when undoing moves
            # piece.clear_position()
            
            return piece
        
        return None

    def peek(self):
        '''
            Returns the piece from this tile.
        '''
        return self.piece

    def reward(self):
        '''
            Returns the reward of this tile.
        '''
        return self.piece.reward if self.occupied() else 0

    def vector(self):
        vector = self.piece.vector() if self.occupied() else np.zeros((1, 12))
        return vector

    def copy(self):
        return AbstractChessTile(self.color, self.position, self.piece.copy() if self.piece else None)

    def __str__(self):
        return str(self.peek()) if self.occupied() else EMPTY_TILE

    def __hash__(self):
        return hash(self.piece)