from games.gardner.GardnerMiniChessLogic import Board
import numpy as np


class DarkBoard(Board):
    def add_darkness(self, player):
        legal_actions = self.get_legal_moves(player)

        flat_pieces = [item for sublist in self.pieces for item in sublist]

        flat_dark = np.zeros(len(flat_pieces))


        # let us see all of our own pieces
        for i in range(len(flat_pieces)):
            if flat_pieces[i] * player > 0:
                flat_dark[i] = flat_pieces[i] * player

        for _,_,j in legal_actions:
            flat_dark[j] = flat_pieces[j] * player

        self.pieces = np.array(flat_dark).reshape((self.n+4,self.n+2))

        return self.pieces_without_padding()
            

