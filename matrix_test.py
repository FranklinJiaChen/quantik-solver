import numpy as np

# Original 4x4 matrix 'board'
board = [
    ['a', 'b', 'c', 'd'],
    ['e', 'f', 'g', 'h'],
    ['i', 'j', 'k', 'l'],
    ['m', 'n', 'o', 'p']
]

# Function to find the position of '*'
def find_cell(matrix):
    for r in range(len(matrix)):
        for c in range(len(matrix[0])):
            if '*' in matrix[r][c]:
                return (c, r)
    return None

# Define the 8 transformations/functions
d4_functions = [
    lambda matrix: matrix,                                # Identity function
    lambda matrix: np.rot90(matrix),                      # Rotate 90 degrees
    lambda matrix: np.rot90(matrix, 2),                   # Rotate 180 degrees
    lambda matrix: np.rot90(matrix, 3),                   # Rotate 270 degrees
    lambda matrix: np.fliplr(matrix),                     # Reflect across vertical axis
    lambda matrix: np.flipud(matrix),                     # Reflect across horizontal axis
    lambda matrix: np.rot90(np.fliplr(matrix)),           # Reflect across top-left to bottom-right diagonal
    lambda matrix: np.rot90(np.flipud(matrix))            # Reflect across bottom-left to top-right diagonal
]

# Apply each transformation and find the new position of '*'
for i, func in enumerate(d4_functions):
    board_np = np.array(board, dtype='object')  # Use dtype='object' to handle strings correctly
    board_np[1][2] = board_np[1][2] + '*'

    transformed_matrix = func(board_np)
    new_position = find_cell(transformed_matrix)

    print(f"Function {i + 1}:\nTransformed Matrix:\n{transformed_matrix}\nNew Cell Position: {new_position}\n")
