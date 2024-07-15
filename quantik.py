import numpy as np
import itertools
import mysql.connector
from utils import (matrix_to_flat_list, flat_list_to_matrix,
                   matrix_to_flat_tuple, flat_tuple_to_matrix)

SQL_CONFIG = {
    'user': 'root',
    'host': 'localhost',
    'database': 'quantik',
}

# Symmetries for the game of Quantik, for normalization
# Follows Sudoku symmetries.
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


def is_terminal(board: tuple[int], pos: int) -> bool:
    """
    Given a board state and a new piece position,
    check if the board state is terminal.
    """
    row = pos // 4
    col = pos % 4
    region = get_region(row, col)

    # Check rows
    row_elements = [abs(board[row*4+i]) for i in range(4) if board[row*4+i] != 0]
    if len(set(row_elements)) == 4:
        return True
    # Check columns
    col_elements = [abs(board[i*4+col]) for i in range(4) if board[i*4+col] != 0]
    if len(set(col_elements)) == 4:
        return True
    # Check regions
    region_cells = [board[j*4+k] for j in range(2*(region//2), 2*(region//2) + 2)
                    for k in range(2*(region % 2), 2*(region % 2) + 2) if board[j*4+k] != 0]
    if len(set(region_cells)) == 4:
        return True

    return False



def connect_to_database() -> None:
    """
    Connects to the MySQL database.
    """
    global conn, cursor
    try:
        conn = mysql.connector.connect(**SQL_CONFIG)
        cursor = conn.cursor()
    except mysql.connector.Error as e:
        print(f"Error connecting to MySQL: {e}")
        return

def disconnect_from_database() -> None:
    """
    Disconnects from the MySQL database.
    """
    if 'conn' in locals() and conn.is_connected():
        cursor.close()
        conn.close()
        print('MySQL connection closed')


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
    minimax_count = 0
    def __init__(self,
                 board: list[list[int]] =
                    [[0 for _ in range(4)] for _ in range(4)]) -> None:

        self.board = get_normalized_form(np.array(board))
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

        if len(self.player1_pieces) < len(self.player2_pieces):
            self.current_player = 2
        else:
            self.current_player = 1

        self.move_number = 16 - len(self.player1_pieces) - len(self.player2_pieces)


    def get_moves(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Note: This function has a conditional
               return type (to speed up solving)

        Returns an immediate winning move or all possible moves.

        Returns
        - tuple[np.ndarray, list[np.ndarray]]:
            A tuple containing the immediate winning move
            and all possible moves.

        If no immediate winning move is found,
            the first element of the tuple is None.
        If a winning move is found, the second element of the tuple is None.
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
                    if is_terminal(new_board, pos):
                        return new_board, None
                    moves_set.add(new_board)
                    moves.append(flat_tuple_to_matrix(new_board))

        return None, moves


    def is_terminal(self) -> bool:
        """
        Returns whether the current state is a terminal state.

        Returns:
        - bool: True if the current state is a terminal state,
                False otherwise.
        """
        for i in range(4):
            # Check rows
            if all(num in self.board[i] or -num in self.board[i] for num in range(1, 5)):
                return True

            # Check columns
            if all(num in self.board[:, i] or -num in self.board[:, i] for num in range(1, 5)):
                return True

            # Check regions
            region_cells = [self.board[j, k] for j in range(2*(i//2), 2*(i//2) + 2)
                            for k in range(2*(i % 2), 2*(i % 2) + 2)]
            if all(num in region_cells or -num in region_cells for num in range(1, 5)):
                return True
        return False

    def evaluate_board(self, alpha: int, beta: int) -> int:
        """
        Returns the evaluation of the board state. (uses minimax
        and alpha-beta pruning)

        Returns:
        - int: The evaluation of the board state.
            - If player 1 can force a win, return 1.
            - If player 2 can force a win, return -1.
        """
        Quantik.minimax_count += 1
        print("counter:", Quantik.minimax_count)

        pruned = False
        cursor.execute(f'''
                        SELECT eval FROM quantik
                        WHERE board = "{str(self.board)}"
                        ''')
        board_data = cursor.fetchone()
        if board_data and board_data[0]: return board_data[0]

        self.set_info()

        best_move, moves = self.get_moves()
        if best_move:
            # Base case: current player has an immediate win
            eval = 1 if self.current_player == 1 else -1
        elif not moves:
            # Base case: current player has no moves left and loses
            eval = 1 if self.current_player == 2 else -1
        # minimax with alpha-beta pruning
        else:
            best_score = float('-inf') if self.current_player == 1 else float('inf')
            for move in moves:
                quantik_game = Quantik(move)
                score = quantik_game.evaluate_board(alpha, beta)
                best_score = max(best_score, score) if self.current_player == 1 else min(best_score, score)
                if self.current_player == 1:
                    alpha = max(alpha, score)
                else:
                    beta = min(beta, score)
                # alpha-beta pruning
                if beta <= alpha:
                    pruned = True
                    break
                # break if we find a winning move
                if self.current_player == 1 and best_score == 1:
                    break
                if self.current_player == 2 and best_score == -1:
                    break
            eval = best_score

        print(f"{self.board}")
        if not pruned:
            cursor.execute(f'''
                            INSERT INTO quantik (board, eval, move_number)
                            VALUES ("{str(self.board)}", {eval}, {self.move_number})
                            ''')
            conn.commit()
        return eval

def main():
    connect_to_database()
    game = Quantik()
    game.print_board()
    print(game.evaluate_board(float('-inf'), float('inf')))
    disconnect_from_database()


if __name__ == '__main__':
    main()
