import numpy as np

# converts 4x4 matrix into a flat list
matrix_to_flat_list = lambda matrix: list(matrix.flatten())

# converts flat list into 4x4 matrix
flat_list_to_matrix = lambda flat_tuple: np.array(flat_tuple).reshape((4, 4))

# converts 4x4 matrix into a flat tuple
matrix_to_flat_tuple = lambda matrix: tuple(matrix.flatten())

# converts flat tuple into 4x4 matrix
flat_tuple_to_matrix = lambda flat_tuple: np.array(flat_tuple).reshape((4, 4))