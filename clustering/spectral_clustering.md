
# Spectral Clustering
Here I will derive the mathematical basics of why does spectral clustering work. I will break them into four parts. The first three parts will lay the required groundwork for the mathematics behind spectral clustering. The final part will be piecing everything together and show that why that spectral clustering works as intended.

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
Lagrange multiplier is frequently used in classical mechanics to solve function optimization under constraints. Lagrangian mechanic is often used in non-relativistic quantum mechanics for particle physics, however that would require knowledge in path integral. In this section, I am sticking to the plain old Lagrange multipler to solve a simple constraint problem as an example. 

Suppose we want to minimize or maximize a function $$f(x, y, z)$$ while subjecting to a constraint function $$g(x, y, z) = k$$. The easier example one can think of is that you want to a fence around your house, but the constraint is that you have limited materials from Home Depot. You want to maximize the area enclosed by your fence. Then you can use Lagrange Multiplier to perform the calculation and find the opitmal solution. 

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


## Part 3 - Minimize $x^{T}Ax$
The objective here is to combine what we know from Part 1 and Part 2 to achieve the following:

$$
argmin_{\vec{x}} \; x^{T}Ax
$$

We want to minimize the expression `x * A.T * x` under the following constraints: 

$$
\text{x is normalized} \quad x^{T}x = 1
$$

$$
\text{symmetric} \quad A = A^{T}
$$

$$
\text{A is positive semidefinite} \quad x^{T}Ax \geq 0
$$

Being positive semidefinite is an important quality, because if a matrix is definite or semidefinite positive, the vector, at which derivative of the expression is zero, has to be the solution for minimization. Now we have our constraints, we are ready to use Lagrange multiplier to minimize this expression.

$$
\mathcal{L} = x^{T}Ax - \lambda\left(x^{T}x - 1\right)
$$

We will make an important assumption here, that `A` is a symmetric matrix. You will see that this is indeed the case later on. 

$$
\frac{\partial \mathcal{L}}{\partial x} = 2Ax - 2\lambda x = 0
$$

$$
\frac{\partial \mathcal{L}}{\partial \lambda} = 1 - x^{T}x = 0
$$

Now solve for $$\vec{x}$$ and I will begin to use the vector notation here in case we forget that $x$ is a vector, and it has always been a vector. I omitted the vector symbol to type less LaTeX code on my end but I must include it here to illustrate a point:

$$
A\vec{x} = \lambda\vec{x}
$$

The constraint equation gave us that

$$
x^{T}x = 1
$$

So what does this mean? It means that if you want to minimize the expression $$x^{T}Ax$$, $$\vec{x}$$ must be the eigenvectors of $$A$$! Here are couple important properties:

* $$A$$ is positive if all eigenvalues are positive.
* $$A$$ is semidefinite positive if all eigenvalues are either positive or zero.
* All eigenvectors are the same size, they have a norm equal to 1.
* The eigenvector corresponding to the smallest eigenvalue will give you the smallest possible value of $$A\vec{x}$$
* In converse, eigenvector corresponding to the biggest eigenvalue will give you the maximum of $$A\vec{x}$$. However I am not 100% sure of this point, I need to run couple tests to verify it. 


