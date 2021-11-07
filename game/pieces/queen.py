import numpy as np
from game.abstract.piece import AbstractChessPiece, PieceColor
from game.abstract.resources import WHITE_QUEEN, BLACK_QUEEN

class Queen(AbstractChessPiece):
    def __init__(self, color: PieceColor, position: tuple, value: int) -> None:
        super().__init__(color, position, value)

    def _onehot(self):
        return np.array([0, 0, 0, 0, 1, 0])

    def __str__(self):
        return WHITE_QUEEN if self.color == PieceColor.WHITE else BLACK_QUEEN