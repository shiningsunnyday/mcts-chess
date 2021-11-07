from games.gardner.GardnerMiniChessLogic import Board
from itertools import count
import numpy as np

class MonochromaticChessBoard(Board):
    def _get_legal_moves(self,player):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        flat_pieces = [item for sublist in self.pieces for item in sublist]
        for i, p in enumerate(flat_pieces):
            if p <= 0 or abs(p) == Board.INF: continue
            for d in self.directions[p]:
                for j in count(i+d, d):
                    q = flat_pieces[j]
                    # Stay inside the board, and off friendly pieces
                    if q > 0 or abs(q) == Board.INF: break


                    # Pawn move, double move and capture
                    if p == Board.PAWN and d in (self.north, self.north+self.north) and q != Board.BLANK: break
                    if p == Board.PAWN and d == self.north+self.north and (i < self.bottom_left+self.north or flat_pieces[i+self.north] != Board.BLANK): break
                    if p == Board.PAWN and d in (self.north+self.west, self.north+self.east) and q == Board.BLANK and j not in (self.ep, self.kp): break
                    # Move it
                    if (j - i) % 2 == 0: # TODO
                        yield (abs(p), i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in [Board.PAWN,Board.KNIGHT,Board.KING] or q < 0: break