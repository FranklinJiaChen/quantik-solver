import numpy as np
import time

# Create two example matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[1, 2], [3, 5]])





# Measure time for list comparison
start_list = time.time()

for _ in range(1000000):
    flat_list1 = list(matrix1.flatten())
    flat_list2 = list(matrix2.flatten())
    _ = flat_list1 < flat_list2
end_list = time.time()

# Measure time for tuple comparison
start_tuple = time.time()
for _ in range(1000000):
    flat_tuple1 = tuple(matrix1.flatten())
    flat_tuple2 = tuple(matrix2.flatten())
    _ = flat_tuple1 < flat_tuple2
end_tuple = time.time()

print(flat_tuple1)
# Print results
print(f"List comparison time: {end_list - start_list:.5f} seconds")
print(f"Tuple comparison time: {end_tuple - start_tuple:.5f} seconds")



# create 4x4 matrix
matrix3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
flat_tuple3 = tuple(matrix3.flatten())
print(flat_tuple3)

matrix3 = np.array(flat_tuple3).reshape((4, 4))
print(matrix3)
