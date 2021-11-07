from games.gardner.GardnerMiniChessLogic import Board
import numpy as np

class RifleBoard(Board):
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

        is_capture = (q != 0)

        # in rifle chess, we do not move on captures
        if is_capture:
            board = put(board, j, Board.BLANK)
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