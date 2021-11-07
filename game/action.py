from typing import List, Tuple, Union
from game.abstract.board import AbstractChessBoard
from game.abstract.piece import AbstractChessPiece, PieceColor
from game.pieces import Pawn, Knight, Bishop, Rook, Queen, King
from game.abstract.action import AbstractActionFlags, AbstractChessAction, AbstractChessActionVisitor, visitor
from game.action_reference import ID_TO_ACTION, ACTION_TO_ID, ID_TO_ACTION_BY_TILE, ID_TO_ACTION_BY_TILE

import numpy as np

LEN_ACTION_SPACE = 1225

class GardnerChessAction(AbstractChessAction):

    def __init__(self, agent: AbstractChessPiece, from_pos: tuple, to_pos: tuple, captured_piece: AbstractChessPiece = None, modifier_flags: List[AbstractActionFlags] = None):
        super().__init__(agent, from_pos, to_pos, captured_piece, modifier_flags)

    def encode(self) -> np.array:
        idx = self.idx()

        onehot = np.zeros(LEN_ACTION_SPACE)
        onehot[idx] = 1

        return onehot

    def idx(self) -> int:
        modifier = (-1 if self.agent.color == PieceColor.WHITE else 1, 1 if self.agent.color == PieceColor.WHITE else -1)
        # modifier = (1, 1)

        delta = (
            modifier[0] * (self.to_pos[0] - self.from_pos[0]),
            modifier[1] * (self.to_pos[1] - self.from_pos[1])
            )

        action_tuple = tuple()

        # check for underpromotion flags
        if AbstractActionFlags.PROMOTE_BISHOP in self.modifier_flags:
            action_tuple = (self.from_pos, (delta, 'bishop'))
        elif AbstractActionFlags.PROMOTE_KNIGHT in self.modifier_flags:
            action_tuple = (self.from_pos, (delta, 'knight'))
        elif AbstractActionFlags.PROMOTE_ROOK in self.modifier_flags:
            action_tuple = (self.from_pos, (delta, 'rook'))
        else: # else just a normal action tuple
            action_tuple = (self.from_pos, delta)

        idx = ACTION_TO_ID[action_tuple]
        return idx

    @staticmethod
    def decode(encoding: Union[np.array, int], state_tm1: AbstractChessBoard, should_sanitize=True):
        idx = encoding if type(encoding) == int else np.argmax(encoding)
        
        modifier = (-1 if state_tm1.active_color == PieceColor.WHITE else 1, 1 if state_tm1.active_color == PieceColor.WHITE else -1)

        action = ID_TO_ACTION[idx]

        # action could be ((4, 4), ((1, 1), 'rook'))) or ((4, 1), (-3, -3))

        from_pos = action[0]
        agent = state_tm1.get(from_pos).peek()

        if should_sanitize: assert agent is not None, 'Could not decode action. There exists no piece at {} for board:\n{}'.format(from_pos, state_tm1)

        second_tup = action[1]

        if type(second_tup[1]) == str: # (1, 1), 'rook')
            delta = second_tup[0]
            delta = (modifier[0] * delta[0], modifier[1] * delta[1])
            underpromote = second_tup[1]

            to_pos = (from_pos[0] + delta[0], from_pos[1] + delta[1])

            captured_piece = state_tm1.get(to_pos).peek()

            modifier_flags = [] if captured_piece is None else [AbstractActionFlags.CAPTURE]

            if underpromote == 'rook':
                modifier_flags.append(AbstractActionFlags.PROMOTE_ROOK)
            elif underpromote == 'bishop':
                modifier_flags.append(AbstractActionFlags.PROMOTE_BISHOP)
            elif underpromote == 'knight':
                modifier_flags.append(AbstractActionFlags.PROMOTE_KNIGHT)
            else:
                raise RuntimeError('Invalid AbstractActionFlag type {}'.format(underpromote))

            return GardnerChessAction(agent, from_pos, to_pos, captured_piece, modifier_flags)
        else: # (-3, -3)
            delta = second_tup
            delta = (modifier[0] * delta[0], modifier[1] * delta[1])
            to_pos = (from_pos[0] + delta[0], from_pos[1] + delta[1])

            captured_piece = state_tm1.get(to_pos).peek()

            modifier_flags = [] if captured_piece is None else [AbstractActionFlags.CAPTURE]

            return GardnerChessAction(agent, from_pos, to_pos, captured_piece, modifier_flags)

    def fliplr(self):
        '''
            Returns
            -------
            Returns this action rotated about the y-axis. E.g. moving up-right diagonal one is not up-left diagonal one.
        '''
        from_row,from_col = self.from_pos
        to_row,to_col = self.to_pos

        return type(self)(self.agent, (from_row, 4 - from_col), (to_row, 4 - to_col), self.captured_piece, self.modifier_flags.copy())

    def flipud(self):
        '''
            Returns
            -------
            Returns this action rotated about the y-axis. E.g. moving up-right diagonal one is not up-left diagonal one.
        '''
        from_row,from_col = self.from_pos
        to_row,to_col = self.to_pos

        return type(self)(self.agent, (4 - from_row, from_col), (4 - to_row, to_col), self.captured_piece, self.modifier_flags.copy())

# TODO check-filtering

class GardnerChessActionVisitor(AbstractChessActionVisitor):
    '''
        All standard chess rules, minus pawn double-move and castling
    '''
    
    @visitor(Pawn)
    def visit(self, piece: AbstractChessPiece, board) -> List[AbstractChessAction]:
        row,col = piece.position
        color = piece.color

        row_dir = 1 if color == PieceColor.BLACK else -1
        col_dir = -1 if color == PieceColor.BLACK else 1

        possible_moves = []

        # standard forward move
        forward_one_pos = (row + row_dir, col)
        possible_moves.extend(self._pawn_move_helper(piece, board, forward_one_pos))

        # forward-left capture
        forward_left_pos = (row + row_dir, col - col_dir)
        possible_moves.extend(self._pawn_move_helper(piece, board, forward_left_pos, True))

        # forward-right capture
        forward_right_pos = (row + row_dir, col + col_dir)
        possible_moves.extend(self._pawn_move_helper(piece, board, forward_right_pos, True))

        return possible_moves
    
    def _pawn_move_helper(self, piece: Pawn, board, new_position: tuple, is_capture = False) -> List[AbstractChessAction]:
        '''
            Helper function for pawn moves.
        '''
        possible_moves = []

        if board.is_valid_position(new_position):

            if (is_capture and board.get(new_position).capturable(piece.color)) or not (is_capture or board.get(new_position).occupied()):

                # check if this is last row
                if new_position[0] in [0, 4]: # if yes, we must promote

                    for flag in [AbstractActionFlags.PROMOTE_QUEEN, AbstractActionFlags.PROMOTE_KNIGHT,
                                    AbstractActionFlags.PROMOTE_BISHOP, AbstractActionFlags.PROMOTE_ROOK]:

                        possible_moves.append(
                            GardnerChessAction(
                                piece,
                                piece.position,
                                new_position,
                                board.get(new_position).peek() if is_capture else None,
                                [flag] + ([AbstractActionFlags.CAPTURE] if is_capture else [])
                            )
                        )

                else: # if no, just normal move
                    possible_moves.append(
                        GardnerChessAction(
                            piece,
                            piece.position,
                            new_position,
                            board.get(new_position).peek() if is_capture else None,
                            [AbstractActionFlags.CAPTURE] if is_capture else []
                        )
                    )

        return possible_moves

    def _piece_move_helper(self, piece: AbstractChessPiece, board, new_pos: Tuple[int, int]) -> List[AbstractChessAction]:
        '''
            Helper function for generic piece moves.
        '''

        if board.is_valid_position(new_pos):
            if board.get(new_pos).occupied():
                if board.get(new_pos).capturable(piece.color):
                    return [GardnerChessAction(
                        piece,
                        piece.position,
                        new_pos,
                        board.get(new_pos).peek(),
                        [AbstractActionFlags.CAPTURE]
                    )]
            else:
                return [GardnerChessAction(
                    piece,
                    piece.position,
                    new_pos,
                    board.get(new_pos).peek(),
                    []
                )]

        return []

    @visitor(Knight)
    def visit(self, piece: AbstractChessPiece, board) -> List[AbstractChessAction]:
        row,col = piece.position
        color = piece.color

        directions = [1, -1]

        possible_moves = []

        for y_dir in directions:
            for x_dir in directions:

                # y major axis
                new_row, new_col = row + (y_dir * 2), col + (x_dir)
                new_pos = (new_row, new_col)

                if board.is_valid_position(new_pos):
                    possible_moves.extend(self._piece_move_helper(piece, board, new_pos))

                    # x major axis
                    new_row, new_col = row + (y_dir), col + (2 * x_dir)
                    new_pos = (new_row, new_col)

                    possible_moves.extend(self._piece_move_helper(piece, board, new_pos))

        return possible_moves

    @visitor(Bishop)
    def visit(self, piece: AbstractChessPiece, board) -> List[AbstractChessAction]:
        return self._bishop_move_helper(piece, board)

    
    def _bishop_move_helper(self, piece: AbstractChessPiece, board, max_dist=4) -> List[AbstractChessAction]:
        '''
            Helper function for bishops (and the bishop functionality of queens).
        '''
        row, col = piece.position

        possible_moves = []

        directions = [1, -1]
        
        for x_dir in directions:
            for y_dir in directions:
                collision = False
                for magnitude in range(1, max_dist + 1): # because we can only ever move between [1,4] tiles
                    if not collision:
                        x_change = x_dir * magnitude
                        y_change = y_dir * magnitude

                        new_row, new_col = row + y_change, col + x_change

                        new_pos = (new_row, new_col)

                        if board.is_valid_position(new_pos):
                            possible_moves.extend(self._piece_move_helper(piece, board, new_pos))

                            if board.get(new_pos).occupied():
                                collision = True

        return possible_moves


    @visitor(Rook)
    def visit(self, piece: AbstractChessPiece, board) -> List[AbstractChessAction]:
        return self._rook_move_helper(piece, board)

    def _rook_move_helper(self, piece: AbstractChessPiece, board, max_dist=4):
        '''
            Helper function for rooks (and the rooks functionality of queens).
        '''
        row, col = piece.position

        possible_moves = []

        directions = [(1, 0), (0, -1), (0, 1), (-1, 0)]

        for direction in directions:
            collision = False
            for magnitude in range(1, max_dist + 1): # because we can only ever move between [1,4] tiles
                if not collision:
                    x_change = direction[1] * magnitude
                    y_change = direction[0] * magnitude

                    new_row, new_col = row + y_change, col + x_change

                    new_pos = (new_row, new_col)

                    if board.is_valid_position(new_pos):
                        possible_moves.extend(self._piece_move_helper(piece, board, new_pos))

                        if board.get(new_pos).occupied():
                            collision = True
        
        return possible_moves

    @visitor(Queen)
    def visit(self, piece: AbstractChessPiece, board) -> List[AbstractChessAction]:
        return self._bishop_move_helper(piece, board) + self._rook_move_helper(piece, board)

    def _king_move_helper(self, piece: AbstractChessPiece, board) -> List[AbstractChessAction]:
        diag_moves = self._bishop_move_helper(piece, board, max_dist=1)
        rook_moves = self._rook_move_helper(piece, board, max_dist=1)

        return diag_moves + rook_moves

    @visitor(King)
    def visit(self, piece: AbstractChessPiece, board) -> List[AbstractChessAction]:
        return self._king_move_helper(piece, board)
