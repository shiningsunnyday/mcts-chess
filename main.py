import random
import chess 
from mcts_state import BoardState
from mcts import mcts

def move(s: str):
    return chess.Move.from_uci(s)

if __name__ == "__main__":
    board = chess.Board()
    mcts = mcts(timeLimit=5000)
    color = False
    while not board.is_game_over():
        move = mcts.search(initialState=BoardState(board.copy(stack=False), color))
        board.push(move)
        if not board.is_game_over():
            board.push(random.choice(list(board.legal_moves)))
        print(board)
