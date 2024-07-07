import numpy as np

def swap_symbols(board: np.ndarray):
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
    largest_positive = 4
    largest_negative = -1

    # Iterate over each element's indices
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            cell = board[row, col]  # Get the current cell value
            if cell not in mapping:
                if cell > 0:
                    mapping[cell] = largest_positive
                    largest_positive -= 1
                else:
                    mapping[cell] = largest_negative
                    largest_negative -= 1
            # Update the cell value in the original board
            board[row, col] = mapping[cell]

    return board

# Original 4x4 matrix 'board'
board = np.array([
    [1, 1, 2, 3],
    [-4, -3, -3, 3],
    [1, -4, 2, -4],
    [0, 4, 0, 0]
], dtype='object')

# Swap the symbols of the matrix
new_board = swap_symbols(board.copy())  # Use .copy() to avoid modifying the original board


# Print the original and new matrix
print("Original Matrix:")
print(board)

print("\nSwapped Matrix:")
print(new_board)
