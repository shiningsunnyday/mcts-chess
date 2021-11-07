
from enum import Enum
import numpy as np

class PieceColor(Enum):
    WHITE = 0
    BLACK = 1

    def invert(self):
        '''
            Inverts this color.

            Returns
            -------
            BLACK if self == BLACK else WHITE
        '''

        return PieceColor.BLACK if self == PieceColor.WHITE else PieceColor.WHITE

    def __str__(self) -> str:
        '''
            Returns
            -------
            'black' if self == BLACK else 'white
        '''

        if self == PieceColor.WHITE:
            return 'white'
        elif self == PieceColor.BLACK:
            return 'black'
        else:
            raise RuntimeError('Invalid PieceColor')

class AbstractChessPiece:
    '''
        Abstract Data Type representing a MiniChess piece.
    '''
    
    def __init__(self, color: PieceColor, position: tuple, value: int = 0) -> None:
        self.color = color
        self.position = position
        self.value = value

    def set_position(self, position: tuple) -> None:
        '''
            Sets the position of this piece to `position`.

            Parameters
            ----------
            position :: tuple : a (row, col) tuple representing the position of this piece.
        '''

        self.position = position

    def clear_position(self) -> None:
        '''
            Sets the position of this piece to (-1, -1)
        '''

        self.position = (-1, -1)

    def _onehot(self) -> np.array:
        '''
            Returns
            -------
            numpy array representing the one-hot encoding of this piece:
            ```
                [1, 0, 0, 0, 0, 0] = Pawn
                [0, 1, 0, 0, 0, 0] = Knight
                [0, 0, 1, 0, 0, 0] = Bishop
                [0, 0, 0, 1, 0, 0] = Rook
                [0, 0, 0, 0, 1, 0] = Queen
                [0, 0, 0, 0, 0, 1] = King
            ```
        '''
        raise NotImplementedError

    def vector(self) -> np.array:
        '''
            Returns
            -------
            numpy array representing two self._onehot() vectors concatenated with each other,
            where the first six places correspond to white pieces and latter six correspond to
            black pieces.

            For example:
            
            `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]` is a black bishop
        '''

        color_mask = np.array([
            [int(self.color == PieceColor.WHITE)],
            [int(self.color == PieceColor.BLACK)]
        ])

        onehots = np.array([
            self._onehot(), self._onehot()
        ])

        vector = np.expand_dims(np.hstack(color_mask * onehots), axis=0)

        return vector

    @property
    def reward(self):
        return self.value if self.color == PieceColor.WHITE else -1 * self.value

    def copy(self):
        return type(self)(self.color, self.position, self.value)

    def __str__(self) -> str:
        return '?'

    def __hash__(self):
        return hash(str(self))