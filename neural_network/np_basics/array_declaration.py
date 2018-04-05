import numpy as np

"""Array

The number of dimensions is the rank of the array, the shape of an array is a tuple of integers giving the size of the
array along each dimension
""""

a = np.array([1, 2, 3]) # Create a rank 1 array

print(type(a)) # Prints "<class 'numpy.ndarray'>"
print(a.shape) # Prints "(3,)"
print(a)       # Prints "[1, 2, 3]"
print(a[0], a[1], a[2]) # Prints "1 2 3"

# Numpy also provides many functions to create arrays

a = np.zeros((2, 2)) # Creates an array of all zeros
print(a)

b = np.ones((1, 2)) # Creates an array of all ones
print(b)

c = np.full((2, 2), 7)
print(c)

d = np.eyes(2)
print(d)

e = np.random.random((2, 2))
print(e)
