from games.gardner.GardnerMiniChessLogic import Board
from itertools import count
import numpy as np


class AtomicBoard(Board):
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

        is_capture = (q != Board.BLANK)

        # in atomic chess, we remove all non-pawn pieces surrounding captured piece
        if is_capture:
            board = put(board, j, Board.BLANK)
            board = put(board, i, Board.BLANK)

            # explode
            for neighbor in [j-8, j-7, j-6, j-1, j+1, j+6, j+7, j+8]:
                if abs(flat_pieces[neighbor]) not in (Board.PAWN, Board.INF):
                    board = put(board, neighbor, Board.BLANK)

            return self.rotate(board)
        else:
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
                    
                    # kings cannot capture in atomic chess
                    if p == Board.KING and q != 0: break

                    # Pawn move, double move and capture
                    if p == Board.PAWN and d in (self.north, self.north+self.north) and q != Board.BLANK: break
                    if p == Board.PAWN and d == self.north+self.north and (i < self.bottom_left+self.north or flat_pieces[i+self.north] != Board.BLANK): break
                    if p == Board.PAWN and d in (self.north+self.west, self.north+self.east) and q == Board.BLANK and j not in (self.ep, self.kp): break
                    # Move it
                    yield (abs(p), i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in [Board.PAWN,Board.KNIGHT,Board.KING] or q < 0: break