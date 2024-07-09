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