import numpy as np
from utils import matrix_to_flat_list, flat_list_to_matrix

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
            cell = board[row, col]  # Get the current cell value
            cell_symbol = abs(cell)  # Get the symbol of the cell
            sign = 1 if cell > 0 else -1  # Get the sign of the cell
            if cell_symbol not in mapping:
                if cell > 0:
                    mapping[cell_symbol] = max(unmapped_symbols)
                    unmapped_symbols.remove(mapping[cell_symbol])
                else:
                    mapping[cell_symbol] = min(unmapped_symbols)
                    unmapped_symbols.remove(mapping[cell_symbol])
            # Update the cell value in the original board
            board[row, col] = mapping[abs(cell)]*sign

    return board

permute_band_symmetry = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: matrix[[2, 3, 0, 1], :]    # Permute Band
]

permute_stack_symmetry = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: matrix[:, [2, 3, 0, 1]]    # Permute Stack
]

permute_row_symmetry = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: matrix[[1, 0, 2, 3], :],   # Permute first band
    lambda matrix: matrix[[0, 1, 3, 2], :],   # Permute second band
    lambda matrix: matrix[[1, 0, 3, 2], :]    # Permute both bands
]

permute_col_symmetry = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: matrix[:, [1, 0, 2, 3]],   # Permute first stack
    lambda matrix: matrix[:, [0, 1, 3, 2]],   # Permute second stack
    lambda matrix: matrix[:, [1, 0, 3, 2]]    # Permute both stacks
]

rotational_symmetry = [
    lambda matrix: matrix,                    # Identity function
    lambda matrix: np.rot90(matrix, 1),       # Rotate 90 degrees
    lambda matrix: np.rot90(matrix, 2),       # Rotate 180 degrees
    lambda matrix: np.rot90(matrix, 3)        # Rotate 270 degrees
]

def get_normalized_from(matrix: np.ndarray) -> np.ndarray:
    """
    Generate all possible representations of the input matrix by applying
    all sudoku symmetries and return the
    lexicographically largest representation.

    Parameters:
    matrix (np.ndarray): The input matrix to be normalized.

    Returns:
    np.ndarray: The matrix corresponding to the normalized form.
    """
    possible_representations  = []
    for permute_band in permute_band_symmetry:
        for permute_stack in permute_stack_symmetry:
            for permute_row in permute_row_symmetry:
                for permute_col in permute_col_symmetry:
                    for rotate in rotational_symmetry:
                        possible_representations.append(matrix_to_flat_list(relabel(permute_band(permute_stack(permute_row(permute_col(rotate(matrix))))))))

    return flat_list_to_matrix(max(possible_representations))

# Original 4x4 matrix 'board'
board = np.array([
    [1, 1, 2, 3],
    [-4, -3, -3, 3],
    [1, -4, 2, -4],
    [0, 4, 0, 0]
], dtype='object')

# Get the normalized form of the matrix
normalized_board = get_normalized_from(board)
print(normalized_board)