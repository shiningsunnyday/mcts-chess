import chess 

def move(s: str):
    return chess.Move.from_uci(s)

if __name__ == "__main__":
    board = chess.Board()
    board.push(move("e2e4"))
    print(board)