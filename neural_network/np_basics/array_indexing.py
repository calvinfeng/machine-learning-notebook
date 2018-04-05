import numpy as np

"""Indexing
"""

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# We are slicing the first two rows and columns 1 and 2
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it will modify the original array
print(a[0, 1]) # => 2
b[0, 0] = 77
print(a[0, 1]) # => 77

# You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the
# original array.

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

row_rank_1 = a[1, :] # Rank 1 view of the second row of a
row_rank_2 = a[1:2, :] # Rank 2 view of the second row of a
print(row_rank_1, row_rank_1.shape)
print(row_rank_2, row_rank_2.shape)

col_rank_1 = a[:, 1]
col_rank_2 = a[:, 1:2]
print(col_rank_1, col_rank_1.shape)
print(col_rank_2, col_rank_2.shape)

"""Datatypes

Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric data types that you can
use to construct arrays
"""
x = np.array([1, 2]) # Let numpy choose the datatype
print(x.dtype)

x = np.array([1.0, 2.0]) # Let numpy choose the datatype
print(x.dtype)

x = np.array([1, 2], dtype=np.int64) # Forces a particular datatype
print(x.dtype)
