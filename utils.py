import numpy as np 

PIECES = 'P R N B Q K'
ALL_PIECES = PIECES.split(' ') + PIECES.lower().split(' ')

def bit_mask(board):
    if not isinstance(board, str): board = str(board)
    board = [x.split(' ') for x in board.split('\n')]
    np_board = np.array(board)
    return np.array([p == np_board for p in ALL_PIECES], dtype=int)

if __name__ == "__main__":
    board = 'r n b q k b n r\np p p p p p p p\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\nP P P P P P P P\nR N B Q K B N R'
    print(bit_mask(board))