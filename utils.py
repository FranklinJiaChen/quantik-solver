import numpy as np

# converts 4x4 into a flat list
matrix_to_flat_list = lambda matrix: list(matrix.flatten())

# converts flat list into 4x4 matrix
flat_list_to_matrix = lambda flat_tuple: np.array(flat_tuple).reshape((4, 4))
