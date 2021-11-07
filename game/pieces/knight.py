from game.abstract.resources import BLACK_KNIGHT, WHITE_KNIGHT
import numpy as np
from game.abstract.piece import AbstractChessPiece, PieceColor

class Knight(AbstractChessPiece):
    def __init__(self, color: PieceColor, position: tuple, value: int) -> None:
        super().__init__(color, position, value)

    def _onehot(self):
        return np.array([0, 1, 0, 0, 0, 0])

    def __str__(self):
        return WHITE_KNIGHT if self.color == PieceColor.WHITE else BLACK_KNIGHT