import numpy as np

"""Array Broadcast
"""
# Suppose that we want to add a constant vector to each row of a matrix
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([1, 0, 1])
y = np.empty_like(x) # Create an empty matrix with same shape as x

print(y)
for i in range(3):
    y[i, :] = x[i, :] + v
print(y)

# However if matrix x is very large, it is inefficient to iterate through every row
# We can stack v together and create a new matrix and then perform element wise addition
vv = np.tile(v, (3, 1))
print(vv)
y = x + vv
print(y)

# Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v!!!
y = x + v
print(y)

# ## More examples ##
v = np.array([1, 2, 3])
w = np.array([4, 5])
# To compute an outer product, we need to reshape v to be a column vector of 3 by 1
print(np.reshape(v, (3, 1)) * w)

x = np.array([[1, 2, 3], [4, 5, 6]])
print(x + v)

# Multiply a matrix by a constant
print(x * 2)
