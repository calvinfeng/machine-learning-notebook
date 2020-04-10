
# Spectral Clustering

Here I will derive the mathematical basics of why does spectral clustering work. I will break them
into four parts. The first three parts will lay the required groundwork for the mathematics behind
spectral clustering. The final part will be piecing everything together and show that why that
spectral clustering works as intended.

## Part 1 - Vector Derivative of a Matrix

Given matrices `A` and `x`, how to compute the following?

$$
\frac{\partial}{\partial x} x^{T} A x
$$

For example:

$$
x = [x_{0}, x_{1}]
$$

$$
A = \begin{bmatrix} a_{00} & a_{01} \\ a_{10} & a_{11} \end{bmatrix}
$$

Multiply them out and get the following expression:

$$
x^{T}Ax = a_{00}x_{0}^{2} + a_{01}x_{0}x_{1} + a_{10}x_{0}x_{1} + a_{11}x_{1}^{2}
$$

Think of taking derivative of `x.T * A * x` with respect to a vector as an element wise operation:

$$
\begin{aligned}
\frac{\partial}{\partial \vec{x}} x^{T}Ax &= \begin{bmatrix} \frac{\partial}{\partial x_{0}} \\ \frac{\partial}{\partial x_{1}} \end{bmatrix} x^{T}Ax \\
&= \begin{bmatrix}
2a_{00}x_{0} + a_{01}x_{1} + a_{10}x_{1} \\
a_{01}x_{0} + a_{10}x_{0} + 2a_{11}x_{1}
\end{bmatrix}
\end{aligned}
$$

Which is equivalent to:

$$
\begin{bmatrix}
2a_{00} & a_{01} + a_{10} \\
a_{01} + a_{10} & 2a_{11}
\end{bmatrix}
\begin{bmatrix} x_{0} \\ x_{1} \end{bmatrix}
= \left( A + A^{T} \right) \vec{x}
$$

Thus, the answer is:

$$
\frac{\partial}{\partial x} x^{T} A x = \left( A + A^{T} \right) \vec{x}
$$

## Part 2 - Lagrange Multiplier

Lagrange multiplier is frequently used in classical mechanics to solve function optimization under
constraints. Lagrangian mechanic is often used in non-relativistic quantum mechanics for particle
physics, however that would require knowledge in path integral. In this section, I am sticking to
the plain old Lagrange multipler to solve a simple constraint problem as an example.

Suppose we want to minimize or maximize a function `f(x, y, z)` while subjecting to a constraint
function `g(x, y, z) = k`. The easier example one can think of is that you want to a fence around
your house, but the constraint is that you have limited materials from Home Depot. You want to
maximize the area enclosed by your fence. Then you can use Lagrange Multiplier to perform the
calculation and find the opitmal solution.

### Formulation

We define Lagrangian as follows:

$$
\mathcal{L} = f(x, y, z) - \lambda\left(g(x, y, z) - k\right)
$$

The objective is to solve the following function, which I forgot what was the formal name for it:

$$
\nabla_{x, y, z, \lambda} \mathcal{L}(x, y, z, \lambda) = \vec{0}
$$

We will have four equations and four unknowns.

### Example

Let's use a simple 2 dimensional example with only two variables, `x` and `y`:

$$
f(x, y) = 6x^{2} + 12y^{2}
$$

$$
g(x, y) = x + y = 90
$$

Then the Lagrangian for this example is:

$$
\mathcal{L} = 6x^{2} + 12y^{2} - \lambda\left(x + y - 90\right)
$$

Compute gradient for our Lagrangian and set it to equal to zero:

$$
\nabla_{x, y, \lambda} = \begin{bmatrix}
12x - \lambda \\
24y - \lambda \\
90 - x - y
\end{bmatrix} = \begin{bmatrix}
0 \\
0 \\
0
\end{bmatrix}
$$

Then the solution is clear:

$$
\begin{bmatrix} x \\ y \\ \lambda \end{bmatrix}=
\begin{bmatrix} 60 \\ 30 \\ 720 \end{bmatrix}
$$

## Part 3 - Minimize $$x^{T}Ax$$

The objective here is to combine what we know from Part 1 and Part 2 to achieve the following.

$$
argmin_{\vec{x}} \; x^{T}Ax
$$

We want to minimize the expression `x * A.T * x` under the following constraints.

$$
\text{x is normalized} \quad x^{T}x = 1
$$

$$
\text{symmetric} \quad A = A^{T}
$$

$$
\text{A is positive semidefinite} \quad x^{T}Ax \geq 0
$$

Being positive semidefinite is an important quality, because if a matrix is definite or semidefinite
positive, the vector, at which derivative of the expression is zero, has to be the solution for
minimization. Now we have our constraints, we are ready to use Lagrange multiplier to minimize this
expression.

$$
\mathcal{L} = x^{T}Ax - \lambda\left(x^{T}x - 1\right)
$$

We will make an important assumption here, that `A` is a symmetric matrix. You will see that this is
indeed the case later on.

$$
\frac{\partial \mathcal{L}}{\partial x} = 2Ax - 2\lambda x = 0
$$

$$
\frac{\partial \mathcal{L}}{\partial \lambda} = 1 - x^{T}x = 0
$$

Now solve for `x`. I will begin to use the vector notation here in case we forget that `x` is a
vector, and it has always been a vector. I omitted the vector symbol to type less LaTeX code on my
end but I must include it here to illustrate a point.

$$
A\vec{x} = \lambda\vec{x}
$$

The constraint equation gave us that

$$
x^{T}x = 1
$$

So what does this mean? It means that if you want to minimize the expression $$x^{T}Ax$$, `x` must
be the eigenvectors of `A`! Here are couple important properties:

* `A` is positive if all eigenvalues are positive.
* `A` is semidefinite positive if all eigenvalues are either positive or zero.
* All eigenvectors are the same size, they have a norm equal to 1.
* The eigenvector corresponding to the smallest eigenvalue will give you the smallest possible value
   of $$A\vec{x}$$
* In converse, eigenvector corresponding to the biggest eigenvalue will give you the maximum of
  $$A\vec{x}$$. However I am not 100% sure of this point, I need to run couple tests to verify it.

## Part 4 - Similarity Graph

### Adjacency Matrix

Suppose we are given a set of four vertices or nodes `V = {v1, v2, v3, v4}` and we want to construct
a adjacency matrix as follows:

$$
W_{V} = \begin{bmatrix}
5 && 4 && 4 && 0 \\
4 && 5 && 4 && 0 \\
4 && 4 && 5 && 1 \\
0 && 0 && 1 && 5
\end{bmatrix}
$$

Each element of `W` represents the connectivity strength between vertix `i` and vertix `j`. For
example, `W[0][0]` is the connection strength between vertix 0 and itself (we can assume that 5 is
the maximum connectivity strength.) A zero value can represent that there is no connection. So far
the numbers in the adjacency matrix seems something arbitrary because it is. In general, we should
use a Gaussian Kernel to fill in the connectivity strength.

$$
W_{i, j} = exp\left( \frac{ -1 * \lVert x_{i} - x_{j} \rVert^{2}}{2 * \sigma^{2}} \right)
$$

### Degree Matrix

The degree of a vertex represents its total connectivity.

$$
d_{i} = \Sigma_{j=0}^{N} W_{i, j}
$$

We can then construct a degree matrix as follows.

$$
D_{V} = \begin{bmatrix}
d_{0} && 0 && 0 && 0 \\
0 && d_{1} && 0 && 0 \\
0 && 0 && d_{2} && 0 \\
0 && 0 && 0 && d_{3}
\end{bmatrix}
$$

### Minimizing Relationship

Let's define `A` as a subset of `V` such that $$A \subset V$$. Also define $$\overline{A}$$ to be a
set of vertices that are not in the set `A`. Then we can say that:

$$
A \cup \overline{A} = V
$$

Let's use a concrete example, this is a different example from above. Suppose we are given 6
vertices and here is their coordinates.

```python
import matplotlib.pyplot as plt
%matplotlib inline

Vx = [0, 0, 1, 5, 6, 6]
Vy = [0, 1, 0, 2, 2, 3]

plt.scatter(Vx, Vy)
plt.show()
```

Visually speaking, it is very clear to us that these points can be grouped into two clusters, We can
say that everything in the bottom-left group belongs to the set `A`.

$$
A = \{\{0, 0\}, \{0, 1\}, \{1, 0\}\}
$$

And everything in the upper-right group belongs to the set of `not A`.

$$
\overline{A} = \{\{5, 2\}, \{6, 2\}, \{6, 3\}\}
$$

We can designate a feature vector for these vertices, and this feature vector represents whether a
vertex is in the set `A` or not using 1 to indicate positive and 0 to indicate negative.

$$
f_{A} = \begin{bmatrix}
1 \\ 1 \\ 1 \\ 0 \\ 0 \\ 0
\end{bmatrix}
$$

Now the natural question to ask is, how does this work if we have more than 2 clusters? Suppose we
want 6 clusters, then we simply create 6 feature vectors each represents one of the groups
`A, B, C, D, E, F`.

$$
f_{A}, f_{B}, f_{C}, f_{D}, f_{E}, f_{F} =
\begin{bmatrix}
1 \\ 0 \\ 0 \\ 0 \\ 0 \\ 0
\end{bmatrix}
\begin{bmatrix}
0 \\ 1 \\ 0 \\ 0 \\ 0 \\ 0
\end{bmatrix}
\begin{bmatrix}
0 \\ 0 \\ 1 \\ 0 \\ 0 \\ 0
\end{bmatrix}
\begin{bmatrix}
0 \\ 0 \\ 0 \\ 1 \\ 0 \\ 0
\end{bmatrix}
\begin{bmatrix}
0 \\ 0 \\ 0 \\ 0 \\ 1 \\ 0
\end{bmatrix}
\begin{bmatrix}
0 \\ 0 \\ 0 \\ 0 \\ 0 \\ 1
\end{bmatrix}
$$

Now we can say that the point of spectral clustering is to find feature vectors such that the
relationship between points in set and points not in set are minimized. We can declare the following
equation as our relationship.

$$
R = \Sigma_{i} \Sigma_{j} W_{i, j} (f_{i} - f_{j})^{2}
$$

Our objective is to find the correct feature vector(s) such that when two vertices share strong
connection should be categorized into the same cluster, which in turn minimize the relationship
expression. There exissts many solutions to this minimization as we shall see soon.

$$
argmin_{f} \; \Sigma_{i, j} W_{i, j} (f_{i} - f_{j})^{2}
$$

## Final Derivation

$$
\begin{aligned}
\Sigma_{i, j} W_{i, j} (f_{i} - f_{j})^{2}
&= \Sigma_{i, j} W_{i, j} (f_{i}^{2} - 2f_{i}f_{j} + f_{j}^{2})\\
&= \Sigma_{i, j}W_{i, j}f_{i}^2 + 2\Sigma_{i, j}W_{i, j}f_{i}f_{j} + \Sigma_{i, j}W_{i, j}f_{j}^{2}
\end{aligned}
$$

Since `W` is symmetric and looping through `i` is the same as looping through `j`, we can declare
that the first term and third term of the above equation are equivalent.

$$
\begin{aligned}
\Sigma_{i,j}W_{i, j}f_{i}^{2}
&= \Sigma_{i}(W_{i, 0}f_{i}^{2} + W_{i, 1}f_{i}^{2} + W_{i, 2}f_{i}^{2} + ...) \\
&= \Sigma_{i}(W_{i, 0} + W_{i, 1} + W_{i, 2} + ...) \cdot f_{i}^{2} \\
&= \Sigma_{i}d_{i}f_{i}^{2} \\
&= \vec{f}^{T}D\vec{f}
\end{aligned}
$$

Now we turn to the second term and see how to simplify it.

$$
\begin{aligned}
\Sigma_{i, j} W_{i, j} f_{i} f_{j}
&= \Sigma_{i} f_{i} \Sigma_{j} W_{i, j} f_{j} \\\\
&= \Sigma_{i} f_{i} \begin{bmatrix} W_{i, 0} & W_{i, 1} & W_{i, 2} & ... \end{bmatrix}
\begin{bmatrix} f_{0} \\ f_{1} \\ f_{2} \\ ... \end{bmatrix} \\\\
&= \Sigma_{i} f_{i} \begin{bmatrix} W_{i, 0} & W_{i, 1} & W_{i, 2} & ... \end{bmatrix} \vec{f} \\\\
&= \left(f_{0} \begin{bmatrix} W_{0, 0} & W_{0, 1} & W_{0, 2} & ... \end{bmatrix} + f_{1} \begin{bmatrix} W_{1, 0} & W_{1, 1} & W_{1, 2} & ... \end{bmatrix} + f_{2} \begin{bmatrix} W_{2, 0} & W_{2, 1} & W_{2, 2} & ... \end{bmatrix} + ...\right)\;\vec{f}\\\\
&= \begin{bmatrix} f_{0} & f_{1} & f_{2} ... \end{bmatrix}
\begin{bmatrix} W_{0,0} & W_{0,1} & W_{0,2} & ... \\
W_{1,0} & W_{1,1} & W_{1,2} & ... \\
W_{2,0} & W_{2,1} & ...     & ... \\
\end{bmatrix} \vec{f} \\\\
&= \vec{f}^{T}W\vec{f}
\end{aligned}
$$

Therefore,

$$
\begin{aligned}
\Sigma_{i, j} W_{i, j} (f_{i} - f_{j})^{2}
&= \Sigma_{i, j}W_{i, j}f_{i}^2 + 2\Sigma_{i, j}W_{i, j}f_{i}f_{j} + \Sigma_{i, j}W_{i, j}f_{j}^{2} \\
&= \vec{f}^{T}D\vec{f} - 2\vec{f}^{T}W\vec{f} + \vec{f}^{T}D\vec{f} \\
&= 2\left( \vec{f}^{T}D\vec{f} - \vec{f}^{T}W\vec{f} \right) \\
&= 2\vec{f}^{T} (D - W)\vec{f}
\end{aligned}
$$

Usually we call the term degree matrix minus adjacency matrix, the **Laplacian**. We can EASILY
minimize the expression based on what we derived in **Part 3**. In conclusion, if we want to find
the solutions (vectors `f`) such that it minimizes the connectivity and clustering relationship, we
find the least dominant eigenvectors of the **Laplacian** matrix. This was mindblowing to me when I
first learned about it.

$$
argmin_{f} \; \vec{f}^{T} L \; \vec{f}
$$
