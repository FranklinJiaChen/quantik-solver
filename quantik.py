import numpy as np
from utils import matrix_to_flat_list, flat_list_to_matrix

# region as a function
def get_region(row, col):
    return (row // 2) * 2 + col // 2

class Quantik:
    """
    A class representing the Quantik board game.

    Attributes:
    - board (np.ndarray):
    A 4x4 numpy array representing the game board.

    Cell values are integers in range [-4, 4]
    Where 0 represents an empty cell,
          positive integers represent player 1's pieces, and
          negative integers represent player 2's pieces.

    - current_player (int):
    An integer representing the current player (1 or 2).

    - player1_pieces (list):
    A list of integers representing the pieces held by player 1.

    - player2_pieces (list):
    A list of integers representing the pieces held by player 2.

    Methods:
    - print_board(): Prints the current state of the board.
    """
    def __init__(self) -> None:
        self.board = [[0 for _ in range(4)] for _ in range(4)]
        # self.board = [[0, 0, 0, 0],
        #               [0, 0, 0, 0],
        #               [0, 0, 0, 0],
        #               [0, 0, 0, 0]]

        self.board = np.array(self.board)
        self.current_player = 1  # Player 1 starts first
        self.player1_pieces = [1, 2, 3, 4] * 2
        self.player2_pieces = [-1, -2, -3, -4] * 2


    def print_board(self) -> None:
        """
        Prints the current state of the board.
        """
        for row in self.board:
            print(' '.join(map(lambda x: str(x).rjust(-2), row)))


    def set_info(self) -> None:
        """
        sets the current player, player1 pieces, player2 pieces
        based on the board state
        """
        self.player1_pieces = [1, 2, 3, 4] * 2
        self.player2_pieces = [-1, -2, -3, -4] * 2

        for row in self.board:
            for cell in row:
                if cell > 0:
                    self.player1_pieces.remove(cell)
                elif cell < 0:
                    self.player2_pieces.remove(cell)

        if len(self.player1_pieces) > len(self.player2_pieces):
            self.current_player = 2
        else:
            self.current_player = 1


    def get_moves(self) -> list[np.ndarray]:
        """
        Returns a list of valid moves for the current player.
        """
        moves = []
        board = matrix_to_flat_list(self.board)
        empty_spaces = [i for i, x in enumerate(board) if x == 0]


        if self.current_player == 1:
            pieces = set(self.player1_pieces)
        else:
            pieces = set(self.player2_pieces)

        # iterate over each piece
        for piece in pieces:
            possible_positions = empty_spaces.copy()
            opponent_piece = -piece

            # remove possible positions that opponent have in row/col/region
            indices = [i for i, piece in enumerate(board)
                        if piece == opponent_piece]
            for idx in indices:
                row, col = idx // 4, idx % 4
                region = get_region(row, col)
                possible_positions = [pos for pos in possible_positions
                                      if pos // 4 != row and pos % 4 != col
                                      and get_region(pos // 4, pos % 4)
                                           != region]

            # add possible positions to moves
            for pos in possible_positions:
                new_board = board.copy()
                new_board[pos] = piece
                moves.append(flat_list_to_matrix(new_board))


        print(possible_positions)

        return moves

# Example usage:
game = Quantik()
print(game.get_moves())

