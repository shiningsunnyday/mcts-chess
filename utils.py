import numpy as np 
from chess import pgn
import chess   
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import io 

PIECES = 'P R N B Q K'
ALL_PIECES = PIECES.split(' ') + PIECES.lower().split(' ')
PIECE_MAP = {100: 'P', 280: 'N', 320: 'B', 479: 'R', 929: 'Q', 60000: 'K', -100: 'p', -280: 'n', -320: 'b', -479: 'r', -929: 'q', -60000: 'k', 0: '0'}
INVERSE_PIECE_MAP = {v:k for (k,v) in PIECE_MAP.items()}



OUTCOME_EVAL = {"1-0": float("inf"), "0-1": float("-inf"), "1/2-1/2": 0.0}

def np_board_to_str(board):
    return "\n".join([" ".join(b) for b in board])

def convert_bit_mask(b):
    for (i, k) in enumerate(ALL_PIECES):        
        b[i, :, :][b[i, :, :] > 0] = INVERSE_PIECE_MAP[k]
    return np.bitwise_or.reduce(b, axis=0)

def bit_mask(board, piece_map=ALL_PIECES):
    if isinstance(board, str): 
        board = [x.split(' ') for x in board.split('\n')]
        board = np.array(board)
    
    return np.array([p == board for p in piece_map], dtype=int)

def proc_games(games, stockfish):
    boards = np.empty((12 * 8 * 8))
    y = np.empty((1,))

    for g in games:
        game = pgn.read_game(io.StringIO(g))
        while game.next():
            board = game.board()
            stockfish.set_fen_position(board.fen())
            score = stockfish.get_evaluation()['value']
            board = bit_mask(board).flatten()
            boards = np.vstack((boards, board))
            y = np.vstack((y, score))
            game = game.next()
    return boards, y


def convert_mini(board):
    return np.array([[PIECE_MAP[x] for x in y] for y in board])

def extend_mini(board, i, k=0):
    empty = np.chararray((8,8))
    empty[:] = '0'

    if i == 0:
        empty[k:5+k, k:5+k][:5, :5] = board
    elif i == 1:
        empty[k:5+k, -5-k:][:5, :5] = board 
    elif i == 2:
        empty[-5-k:, k:5+k][:5, :5] = board
    else:
        empty[-5-k:, -5-k:][:5, :5] = board
    return empty

def get_fen(board, turn='w'):
    b = '/'.join([''.join(x.decode('utf-8')) for x in board])
    s = ""
    run = 0
    for c in b:
        if c == '0':
            run += 1
        elif run:
            s += str(run)
            run = 0
            s += c
        else:
            s += c
    if run: s += str(run)
    fen = " ".join([s, turn, '-', '-', '0', '1'])
    return fen


def preprocess(turns):
    for (i, (b, pi, w)) in enumerate(turns): 
        if w < 0: b = (-np.array(b)[::-1]).tolist() # flip perspective of players
        opp_turn = 'w' if w < 0 else 'b'
        fen = get_fen(extend_mini(convert_mini(b), 0, 0), opp_turn) 
        board = chess.Board(fen)
        if not board.is_check(): # ignore those in check yet it's the opp move
            yield (b, pi, w)

def postprocess(boards, y):
    classes = (y > 0).astype(int)
    train_boards, test_boards, train_y, test_y = train_test_split(boards, y, test_size=0.2, random_state=0, stratify=classes)
    scaler = MinMaxScaler()

    scaler.fit(train_y)
    train_y = scaler.transform(train_y)
    test_y = scaler.transform(test_y)

    train_boards = np.expand_dims(train_boards, axis=1)
    test_boards = np.expand_dims(test_boards, axis=1)
    
    return train_boards, test_boards, train_y, test_y



def handle_timeout(signum, frame):
    raise TimeoutError()



def evaluate_mini(board, stockfish, turn='w'):
    board = convert_mini(board)
    scores = []
    for i in range(4):
        for k in range(2):
            board_ = extend_mini(board, i, k)
            fen = get_fen(board_, turn)
            print("fen:",fen)
            
            
            stockfish.set_fen_position(fen)
            score = stockfish.get_evaluation()['value']

            
            scores.append(score)
    return np.mean(scores)


if __name__ == "__main__":
    board = 'r n b q k b n r\np p p p p p p p\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\n. . . . . . . .\nP P P P P P P P\nR N B Q K B N R'
    print(bit_mask(board))