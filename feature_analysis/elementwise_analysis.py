import chess

class BoardFilter:

    def __init__(self, fen):
        self._comparison_board: chess.Board = chess.Board(fen)

    def filter(self, board: chess.Board):
        """
        Return true if all the pieces on the initialized comparison board are present and in the same location.

        Note: Currently only analyzes the current player's pieces
        """
        if board.turn != self._comparison_board.turn:
            return False
        for piece in chess.PIECE_TYPES:
            if self._comparison_board.pieces_mask(piece, self._comparison_board.turn) & board.pieces_mask(piece, board.turn) != self._comparison_board.pieces_mask(piece, self._comparison_board.turn):
                return False
        return True


