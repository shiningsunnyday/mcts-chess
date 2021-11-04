import berserk
from stockfish import Stockfish
from chess import pgn
import io 

from utils import *

API_TOKEN = open("lichess.token","r").readline()
session = berserk.TokenSession(API_TOKEN)
client = berserk.Client(session)
games_gen = client.games.export_by_player('LeelaChess',as_pgn=True,max=10)
games = list(games_gen)
boards = np.empty((12 * 8 * 8))
y = np.empty((1,))
stockfish = Stockfish("/usr/local/Cellar/stockfish/14/bin/stockfish")
stockfish.set_depth(2)

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

print(boards.shape, y.shape)