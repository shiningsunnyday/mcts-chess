from game.action import GardnerChessAction, GardnerChessActionVisitor, LEN_ACTION_SPACE
from game.abstract.action import AbstractActionFlags, AbstractChessAction
from typing import List
from game.abstract.piece import AbstractChessPiece, PieceColor
from game.pieces import Pawn, Knight, Bishop, Rook, Queen, King
from game.abstract.board import AbstractChessBoard, AbstractBoardStatus

import numpy as np

PAWN_VALUE   = 100
KNIGHT_VALUE = 305
BISHOP_VALUE = 333
ROOK_VALUE   = 563
QUEEN_VALUE  = 950
KING_VALUE   = 10000

class GardnerChessBoard(AbstractChessBoard):

    def __init__(self, board=None) -> None:
        super().__init__(5)

        if board == None: self._populate_board()
        else: self._board = board

    def _populate_board(self):
        '''
            Places the pieces on the board according to the Gardner MiniChess rules.
        '''

        black_pawns = [Pawn(PieceColor.BLACK, (-1, -1), PAWN_VALUE) for _ in range(5)]
        black_knight = Knight(PieceColor.BLACK, (-1, -1), KNIGHT_VALUE)
        black_bishop = Bishop(PieceColor.BLACK, (-1, -1), BISHOP_VALUE)
        black_rook = Rook(PieceColor.BLACK, (-1, -1), ROOK_VALUE)
        black_queen = Queen(PieceColor.BLACK, (-1, -1), QUEEN_VALUE)
        black_king = King(PieceColor.BLACK, (-1, -1), KING_VALUE)

        for i in range(self.width):
            self.get((1, i)).push(black_pawns[i])

        self.get((0, 0)).push(black_rook)
        self.get((0, 1)).push(black_knight)
        self.get((0, 2)).push(black_bishop)
        self.get((0, 3)).push(black_queen)
        self.get((0, 4)).push(black_king)

        white_pawns = [Pawn(PieceColor.WHITE, (-1, -1), PAWN_VALUE) for _ in range(5)]
        white_knight = Knight(PieceColor.WHITE, (-1, -1), KNIGHT_VALUE)
        white_bishop = Bishop(PieceColor.WHITE, (-1, -1), BISHOP_VALUE)
        white_rook = Rook(PieceColor.WHITE, (-1, -1), ROOK_VALUE)
        white_queen = Queen(PieceColor.WHITE, (-1, -1), QUEEN_VALUE)
        white_king = King(PieceColor.WHITE, (-1, -1), KING_VALUE)

        for i in range(self.width):
            self.get((3, i)).push(white_pawns[i])
        
        self.get((4, 0)).push(white_rook)
        self.get((4, 1)).push(white_knight)
        self.get((4, 2)).push(white_bishop)
        self.get((4, 3)).push(white_queen)
        self.get((4, 4)).push(white_king)

    @staticmethod
    def from_vector(vector):
        '''
            Decodes a Chess Board from a vector.

            Returns
            -------
            AbstractChessBoard that the vector represents.
        '''

        id_to_piece = {
            0: (Pawn, PAWN_VALUE),
            1: (Knight, KNIGHT_VALUE),
            2: (Bishop, BISHOP_VALUE),
            3: (Rook, ROOK_VALUE),
            4: (Queen, QUEEN_VALUE),
            5: (King, KING_VALUE)
        }

        g = GardnerChessBoard()

        for row in range(g.height):
            for col in range(g.width):
                v_i = vector[row][col]

                assert v_i.shape == (12,)
                
                if np.all(v_i == 0):
                    g.get((row,col)).push(None)
                else:
                    argmax = np.argmax(v_i)
                    color = PieceColor.WHITE if argmax < 6 else PieceColor.BLACK
                    Piece,value = id_to_piece[argmax % 6]

                    g.get((row,col)).push(Piece(color, (-1, -1), value))

        return g

    def wipe_board(self):
        '''
            Removes all pieces from this board.
        '''
        for tile in self:
            tile.pop()

    def is_empty(self):
        '''
            Returns
            -------
            True if there are no pieces on the board, False otherwise.
        '''
        for tile in self:
            if tile.peek() != None:
                return False
        return True

    def push(self, action: AbstractChessAction, check_for_check=True):

        from_pos = action.from_pos
        to_pos = action.to_pos

        if check_for_check:
            checking_move, opp_cant_move = self._is_checking_action(action, self.active_color)

            if checking_move: action.modifier_flags.append(AbstractActionFlags.CHECK)
            if checking_move and opp_cant_move: action.modifier_flags.append(AbstractActionFlags.CHECKMATE)

        agent = self.get(from_pos).pop()
        self.get(to_pos).pop()
        self.get(to_pos).push(agent)

        # check for promotions
        if type(agent) == Pawn and agent.position[0] in [0, 4]:
            if AbstractActionFlags.PROMOTE_BISHOP in action.modifier_flags:
                self.get(to_pos).pop()
                self.get(to_pos).push(Bishop(agent.color, to_pos, BISHOP_VALUE))
            elif AbstractActionFlags.PROMOTE_KNIGHT in action.modifier_flags:
                self.get(to_pos).pop()
                self.get(to_pos).push(Knight(agent.color, to_pos, KNIGHT_VALUE))
            elif AbstractActionFlags.PROMOTE_ROOK in action.modifier_flags:
                self.get(to_pos).pop()
                self.get(to_pos).push(Rook(agent.color, to_pos, ROOK_VALUE))
            else:
                self.get(to_pos).pop()
                self.get(to_pos).push(Queen(agent.color, to_pos, QUEEN_VALUE))

        self.move_history.append(action)

        self.active_color = self.active_color.invert()

    def pop(self) -> AbstractChessAction:

        if len(self.move_history) == 0: return None

        action = self.move_history.pop()

        from_pos = action.from_pos
        to_pos = action.to_pos
        agent = action.agent
        captured_piece = action.captured_piece

        self.get(from_pos).pop()
        self.get(from_pos).push(agent)

        self.get(to_pos).pop()
        if captured_piece is not None: self.get(to_pos).push(captured_piece)

        self.active_color = self.active_color.invert()

        return action

    def peek(self):
        return self.move_history[-1]

    def reward(self) -> float:
        reward = 0

        for tile in self:
            reward += tile.reward()

        return reward

    def legal_actions_for_color(self, color: PieceColor, filter_for_check=True) -> List[AbstractChessAction]:

        referee = GardnerChessActionVisitor()
        
        possible_actions = []

        for tile in self:
            piece = tile.peek()

            if piece is not None and piece.color == color:
                possible_actions.extend(referee.visit(piece, self))

        # filter for checks

        if filter_for_check:
            possible_actions = [action for action in possible_actions if not self._leads_to_check(action, color)]

        return possible_actions

    def _leads_to_check(self, action, color):
        '''
            Returns
            -------
            True if this action puts the player that made it in check, false otherwise.
        '''
        
        # simulate this move
        self.push(action, check_for_check=False)

        anti_color = color.invert()

        can_capture_king = False

        for possible_actions in self.legal_actions_for_color(anti_color, filter_for_check=False):
            if AbstractActionFlags.KING_CAPTURE in possible_actions.modifier_flags:
                can_capture_king = True

        self.pop() # undo our move

        return can_capture_king

    def _is_checking_action(self, action, color):
        '''
            Returns
            -------
            tuple of (bool, bool) where
            - the first item is True if this action puts the opponent in check, False otherwise.
            - the second item is True if this action puts the opponent in checkmate, False otherwise.
        '''
        # simulate this move
        self.push(action, check_for_check=False)

        can_capture_king = False

        for possible_actions in self.legal_actions_for_color(color, filter_for_check=False):
            if AbstractActionFlags.KING_CAPTURE in possible_actions.modifier_flags:
                can_capture_king = True

        opponent_cannot_move_next = len(self.legal_actions_for_color(color.invert(), filter_for_check=True)) == 0

        self.pop() # undo our move

        return can_capture_king, opponent_cannot_move_next

    def legal_action_mask(self) -> np.array:

        mask = np.zeros(LEN_ACTION_SPACE)

        for action in self.legal_actions():
            mask += GardnerChessAction.encode(action)

        # assert np.max(mask) == 1, 'max of mask was actually {}'.format(np.max(mask))

        mask[mask > 0] = 1 # correct our mask

        return mask

    def state_vector(self) -> np.array:
        return np.concatenate([
            np.expand_dims(np.concatenate([tile.vector() for tile in row]), axis=0) for row in self._board
        ])

    def canonical_state_vector(self) -> np.array:
        vector = self.state_vector()

        if self.active_color == PieceColor.BLACK:
            # flip board
            vector = np.fliplr(np.flipud(vector))

            # invert all colors
            temp = vector[:,:,:6].copy()
            vector[:,:,:6] = vector[:,:,6:]
            vector[:,:,6:] = temp

        return vector

    @property
    def status(self) -> AbstractBoardStatus:

        # if active color in check...

        opp_can_move = len(self.legal_actions_for_color(self.active_color.invert())) > 0
        opp_actions = self.legal_actions_for_color(self.active_color.invert(), filter_for_check=False)
        opp_checking = False
        for act in opp_actions:
            if AbstractActionFlags.KING_CAPTURE in act.modifier_flags:
                opp_checking = True

        ac_can_move = len(self.legal_actions_for_color(self.active_color)) > 0
        ac_actions = self.legal_actions_for_color(self.active_color, filter_for_check=False)
        ac_checking = False
        for act in ac_actions:
            if AbstractActionFlags.KING_CAPTURE in act.modifier_flags:
                ac_checking = True

        if opp_checking and not ac_can_move:
            return AbstractBoardStatus.BLACK_WIN if self.active_color == PieceColor.WHITE else AbstractBoardStatus.WHITE_WIN
        elif ac_checking and not opp_can_move:
            return AbstractBoardStatus.WHITE_WIN if self.active_color == PieceColor.WHITE else AbstractBoardStatus.BLACK_WIN
        elif (not ((opp_checking or ac_can_move) and (ac_checking or opp_can_move))) or self.has_only_kings:
            return AbstractBoardStatus.DRAW
        else:
            return AbstractBoardStatus.ONGOING

        # OLD IMPLEMENTATION
        # if len(self.move_history) == 0: return AbstractBoardStatus.ONGOING

        # if AbstractActionFlags.CHECKMATE in self.peek().modifier_flags:
        #     return AbstractBoardStatus.WHITE_WIN if self.active_color == PieceColor.BLACK else AbstractBoardStatus.BLACK_WIN
        # elif len(self.legal_actions()) == 0 or self.has_only_kings:
        #     return AbstractBoardStatus.DRAW
        # else:
        #     return AbstractBoardStatus.ONGOING

    def status_for_color(self, color: PieceColor):
        ac = self.active_color
        self.active_color = color
        status = self.status
        self.active_color = ac
        return status

    @property
    def has_only_kings(self) -> bool:
        '''
            Returns
            -------
            True if there are only kings left, false otherwise.
        '''
        
        for tile in self:
            if tile.occupied() and type(tile.peek()) != King:
                return False

        return True

    def copy(self):
        '''
            Returns
            -------
            A deep copy of this board.
        '''
        new_board = type(self)(board=self._board.copy())
        new_board.active_color = self.active_color
        new_board.move_history = self.move_history.copy()

        return new_board