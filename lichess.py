import berserk
from stockfish import Stockfish

from utils import proc_games

if __name__ == "__main__":
    
    API_TOKEN = open("lichess.token", "r").readline().strip('\n')
    session = berserk.TokenSession(API_TOKEN)
    client = berserk.Client(session)
    games_gen = client.games.export_by_player('LeelaChess',as_pgn=True,max=10)
    games = list(games_gen)
    stockfish = Stockfish("/usr/local/Cellar/stockfish/14/bin/stockfish")
    stockfish.set_depth(2)

    boards, y = proc_games(games, stockfish)

    print(boards.shape, y.shape)