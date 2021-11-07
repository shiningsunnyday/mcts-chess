from game.abstract.resources import BLACK_KING, WHITE_KING
import numpy as np
from game.abstract.piece import AbstractChessPiece, PieceColor

class King(AbstractChessPiece):
    def __init__(self, color: PieceColor, position: tuple, value: int) -> None:
        super().__init__(color, position, value)

    def _onehot(self):
        return np.array([0, 0, 0, 0, 0, 1])

    def __str__(self):
        return WHITE_KING if self.color == PieceColor.WHITE else BLACK_KING