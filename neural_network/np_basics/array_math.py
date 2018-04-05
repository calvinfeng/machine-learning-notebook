import numpy as np

"""Array Math
"""
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Elementwise sum
print(x + y)
print(np.add(x, y))

# Elementwise difference
print(x - y)
print(np.subtract(x, y))

# Elementwise product
print(x * y)
print(np.multiply(x, y))

# Elementwise division
print(x / y)
print(np.divide(x, y))

# Elementwise square root
print(np.sqrt(x))

# ## Inner Product ##
w = np.array([9, 10])
v = np.array([11, 12])

# Inner product of two vectors
print(v.dot(w))
print(np.dot(v, w))

# Matrix/vector product
print(x.dot(v))
print(np.dot(x, v))

# Matrix/matrix product
print(x.dot(y))
print(np.dot(x, y))

# We can sum an array
x = np.array([[1, 2], [3, 4]])

print(np.sum(x)) # Compute sum of all elements
print(np.sum(x, axis=0)) # Compute sum of each column => [4, 6]
print(np.sum(x, axis=1)) # Compute sum of each row => [3 7]

# We can do array transpose
x = np.array([[1, 2], [3, 4]])
print(x)
print(x.T)

# Taking transpose of a rank 1 array does nothing
v = np.array([1, 2, 3])
print(v)
print(v.T)
