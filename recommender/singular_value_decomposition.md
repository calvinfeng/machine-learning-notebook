
# Singular Value Decomposition

Using our movie recommendation as an example, we are given a rating matrix $$R$$ and we wish to
perform a singular value decomposition on such matrix, such that

$$
R = U \Sigma M^{T}
$$

The sigma matrix is said to be our diagonal singular matrix, with singular values filling up its
diagonal sorted in decreasing order. The top left corner singular value has the highest value and it
descendes as we move toward the bottom right. The U matrix and M matrix represent the latent
information for each of our users and movies. However, before we dive too deep into the details,
let's do a refresher on singular value decomposition.

## Solving SVD

Using conventional definition, given a matrix A, we want to decompose it into three different
matrices, $$U$$, $$\Sigma$$ and $$V$$. We need to first construct a symmetric matrix of $$A$$
using $$A^{T}A$$.

```python
import numpy as np

# Simple example
A = np.array([[4, 0], [3, -5]], dtype='float')
print A
print A.T.dot(A)
```

    [[ 4.  0.]
     [ 3. -5.]]
    [[ 25. -15.]
     [-15.  25.]]

Then we find the eigenvalues and eigenvectors of this symmetric matrix of $$A$$

```python
eig_vals,eig_vecs =  np.linalg.eig(A.T.dot(A))
```

We will use the square root of its eigenvalues to construct a singular matrix

```python
# Singular values are the sqrt of eigenvalues of (A.T)(A)
s1, s2 = np.sqrt(eig_vals)
S = np.array([[s1, 0], [0, s2]])

# Notice that singular values are sorted from the greatest to least
print S
```

    [[6.32455532 0.        ]
     [0.         3.16227766]]

We can now use the eigenvectors as columns of our V matrix. In this case, numpy has already done it
for us.

```python
V = eig_vecs
print V
```

    [[ 0.70710678  0.70710678]
     [-0.70710678  0.70710678]]

Finally we can solve for U using $$U = AVS^{-1}$$, note that $$S^{-1}$$ is the inverse of $$S$$.

```python
U = A.dot(V).dot(np.linalg.inv(S))
print U
```

    [[ 0.4472136   0.89442719]
     [ 0.89442719 -0.4472136 ]]

SVD is now complete, we can easily verify it by performing the following:

```python
np.isclose(A, U.dot(S).dot(V.T))
```

    array([[ True,  True],
           [ True,  True]])

## Intuition in Recommendation

There are several properties of SVD we should know about.

* It is always possible to decompose a real valued matrix into $$U \Sigma V^{T}$$
* $$U$$, $$\Sigma$$, and $$V$$ are unique
* $$U$$ and $$V$$ are column orthogonal i.e. $$U^{T}U = I$$ and $$V^{T}V = I$$
* $$\Sigma$$ entries are positive and sorted in descending order

Going back to the movie example, imagine that we have 4 movies

* Toy Story
* Finding Nemo
* Braveheart
* Last Samurai

And we have 4 users

* Alice
* Bob
* Calvin
* Debby

We have the following rating matrix from the submitted ratings by each user. And notice that half of
the users likes Pixar animated films a lot while the other half tends to have strong preference
toward historical films.

```python
Rating = np.array([
    [5, 5, 1, 2],
    [1, 1, 5, 5],
    [2, 1, 5, 5],
    [5, 5, 1, 1]
], dtype='float')

# Now let's perform SVD decomposition on this dataset.
eig_vals, eig_vecs =  np.linalg.eig(Rating.T.dot(Rating))
s1, s2, s3, s4 = np.sqrt(eig_vals)

# Let's say we only care about two features, i.e. whether it's animated film or historical film. We will drop the
# other two less important singular values and eigenvectors
S = np.array([[s1, 0], [0, s2]])
M = np.delete(eig_vecs, [2, 3],axis=1)
U = Rating.dot(M).dot(np.linalg.inv(S))
```

```python
S
```

    array([[12.52079729,  0.        ],
           [ 0.        ,  7.53112887]])

```python
M
```

    array([[-0.52034736, -0.46796508],
           [-0.47878871, -0.53010252],
           [-0.47878871,  0.53010252],
           [-0.52034736,  0.46796508]])

```python
U
```

    array([[-0.52034736, -0.46796508],
           [-0.47878871,  0.53010252],
           [-0.52034736,  0.46796508],
           [-0.47878871, -0.53010252]])

Notice that the user matrix, we can clearly see that user 1 and user 4 are similar to each other in
interest and user 2 and 3 are also similar to each other. This is telling us that Alice and Debby
have similar taste in movies and Calvin and Bob share interest in historical drama.

## Low Rank Matrix Factorization

What we did up there is effectively a low rank apprixomation. The original singular matrix had a
rank 4 but we learned that most of the singular values are not really important. We dropped the extra
2 ranks and we can still produce a matrix that similar to the original one.

For example

```python
approximated_rating = U.dot(S).dot(M.T)
print approximated_rating
```

    [[5.03940005 4.98763    1.25114372 1.7408964 ]
     [1.25114372 0.75393781 4.98656303 4.98763   ]
     [1.7408964  1.25114372 4.98763    5.03940005]
     [4.98763    4.98656303 0.75393781 1.25114372]]

```python
# Roud it up
np.round(approximated_rating)
```

    array([[5., 5., 1., 2.],
           [1., 1., 5., 5.],
           [2., 1., 5., 5.],
           [5., 5., 1., 1.]])

Look! It looks just like the original!!!
