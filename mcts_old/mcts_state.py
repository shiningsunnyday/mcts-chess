import random
from mcts import mcts
import chess

class BoardState:
    board : chess.Board
    other_mcts : mcts
    self_mcts : mcts
    color : chess.Color

    def __init__(
        self, 
        board, 
        color, 
        self_mcts = mcts(timeLimit=50), 
        other_mcts = mcts(timeLimit=50)
    ):
        self.board = board
        self.color = color
        self.self_mcts = self_mcts
        self.other_mcts = other_mcts

    def getPossibleActions(self):
        return list(self.board.legal_moves)

    def takeAction(self, action):
        new_board = self.board.copy(stack=False)
        new_board.push(action)
        if not new_board.is_game_over():
        #     other_state = BoardState(new_board, not self.color, self.other_mcts, self.self_mcts)
        #     new_board.push(self.other_mcts.search(initialState=other_state))
            new_board.push(random.choice(list(new_board.legal_moves)))
        return BoardState(new_board, self.color, self.self_mcts, self.other_mcts)

    def isTerminal(self):
        return self.board.is_game_over() or self.board.fullmove_number >= 50

    def getReward(self):
        outcome = self.board.outcome()
        if outcome is None or outcome.winner is None:
            return 0
        return 1 if outcome.winner == self.color else -1