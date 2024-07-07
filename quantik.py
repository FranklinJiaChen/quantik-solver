import numpy as np
import itertools
from utils import (matrix_to_flat_list, flat_list_to_matrix,
                   matrix_to_flat_tuple, flat_tuple_to_matrix)

PERMUTE_BAND_SYMMETRY = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: matrix[[2, 3, 0, 1], :]    # Permute Band
]

PERMUTE_STACK_SYMMETRY = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: matrix[:, [2, 3, 0, 1]]    # Permute Stack
]

PERMUTE_ROW_SYMMETRY = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: matrix[[1, 0, 2, 3], :],   # Permute first band
    lambda matrix: matrix[[0, 1, 3, 2], :],   # Permute second band
    lambda matrix: matrix[[1, 0, 3, 2], :]    # Permute both bands
]

PERMUTE_COL_SYMMETRY = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: matrix[:, [1, 0, 2, 3]],   # Permute first stack
    lambda matrix: matrix[:, [0, 1, 3, 2]],   # Permute second stack
    lambda matrix: matrix[:, [1, 0, 3, 2]]    # Permute both stacks
]

ROTATIONAL_SYMMETRY = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: np.rot90(matrix, 1),       # Rotate 90 degrees
    lambda matrix: np.rot90(matrix, 2),       # Rotate 180 degrees
    lambda matrix: np.rot90(matrix, 3)        # Rotate 270 degrees
]

SYMMETRIES = [
    PERMUTE_BAND_SYMMETRY,
    PERMUTE_STACK_SYMMETRY,
    PERMUTE_ROW_SYMMETRY,
    PERMUTE_COL_SYMMETRY,
    ROTATIONAL_SYMMETRY
]


def get_region(row, col):
    """
    returns the region number of the cell
    Where regions are numbered as follows:
    0 | 1
    -----
    2 | 3

    Args:
    - row (int): row number of the cell 0-3
    - col (int): column number of the cell 0-3

    Returns:
    - int: region number of the cell 0-3
    """
    return (row // 2) * 2 + col // 2


def relabel(board: np.ndarray) -> np.ndarray:
    """
    Given a 4x4 matrix with cells (-4, 4),
    swap the symbols of the matrix such that:
    1. All signs are maintained.
    2. All instances of a symbol are swapped with another symbol.
    3. The new matrix is maximized (lexicographically).

    Args:
    - board (np.ndarray): A 4x4 numpy array representing the game board.

    Returns:
    - np.ndarray: A 4x4 numpy array with the symbols swapped.
    """

    mapping = {0: 0}
    unmapped_symbols = [1, 2, 3, 4]

    # Iterate over each element's indices
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            cell = board[row, col]
            cell_symbol = abs(cell)
            sign = 1 if cell > 0 else -1

            # map the cell symbol such that the new matrix is maximized
            if cell_symbol not in mapping:
                if cell > 0:
                    mapping[cell_symbol] = max(unmapped_symbols)
                    unmapped_symbols.remove(mapping[cell_symbol])
                else:
                    mapping[cell_symbol] = min(unmapped_symbols)
                    unmapped_symbols.remove(mapping[cell_symbol])

            board[row, col] = mapping[abs(cell)]*sign

    return board


def get_normalized_form(matrix: np.ndarray) -> np.ndarray:
    """
    Generate all possible representations of the input matrix by applying
    all sudoku symmetries and return the
    lexicographically largest representation.

    Parameters:
    matrix (np.ndarray): The input matrix to be normalized.

    Returns:
    np.ndarray: The matrix corresponding to the normalized form.
    """
    possible_representations = []
    for transformations in itertools.product(*SYMMETRIES):
        transformed = matrix
        for transform in transformations:
            transformed = transform(transformed)
        relabeled_matrix = relabel(transformed)
        flat_representation = matrix_to_flat_tuple(relabeled_matrix)
        possible_representations.append(flat_representation)

    max_representation = max(possible_representations)
    return flat_list_to_matrix(max_representation)


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
        self.board = [[4, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]]

        self.board = np.array(self.board)

        self.board = get_normalized_form(self.board)
        self.current_player = 2  # Player 1 starts first
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
        moves_set = set() # used to check for duplicates
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
                new_board = matrix_to_flat_tuple(get_normalized_form(flat_list_to_matrix(new_board)))
                if new_board not in moves_set:
                    moves_set.add(new_board)
                    moves.append(flat_tuple_to_matrix(new_board))

        return moves

# Example usage:
game = Quantik()
print(game.get_moves())

