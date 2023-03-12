# KL Divergence

We want to quantify the difference between probability distributions for a given random variable. For example, we'd be
interested in the difference between an actual and observed probability distribution.

> Kullback-Leibler divergence calculates a score that measures the divergence of one probability distribution from another.

## Definition

The KL divergence between two distributions `Q` and `P` is often stated as follows.

$$
\text{KL}(P \| Q)
$$

The operator indicates `P`'s divergence from `Q`, which is not the same as `Q`'s divergence from `P`. Thus, the
operation is not commutative.

$$
\text{KL}(P \| Q) \neq \text{KL}(Q \| P)
$$

KL divergence can be calculated as the negative sum of probability of each event in `P` multiplied by the log of the
probability of the event in `Q` over the probability of the event in `P` .

$$
\text{KL}(P \| Q) = \Sigma_{x \in X} P(x) \cdot log\left( \frac{P(x}{Q(x)} \right)
$$

If there is no divergence at all, the log term becomes 0 and the sum is 0. The log can be base-2 to give units in bits.

If we are attempting to approximate an unknown probability distribution, then the target probability distribution from
data is P and Q is our approximation of the distribution. **In this case, the KL divergence summarizes the number of additional bits required to represent an event from the random variable.**
The better our approximation, the less additional information is required.

## Example

Consider a random variable like drawing a colored marble from a bag. We have 3 colors and two different probability distribution.

```python
events = ['red', 'green', 'blue']
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
```

We can calculate the KL divergence using formula above.

```python
import numpy as np
def kl_divergence(p, q):
    div = 0
    for i in range(3):
        div += p[i] * np.log2(p[i]/q[i])
    return div
```

```python
print("KL(P||Q)", kl_divergence(p, q), "bits")
print("KL(Q||P)", kl_divergence(q, p), "bits")
```

    KL(P||Q) 1.9269790471552186 bits
    KL(Q||P) 2.0216479703638055 bits


## Intuition

[Intuitively Understanding the KL Divergence](https://www.youtube.com/watch?v=SxGYPqCgJWM)

If I have two coins, one fair and one weighted, KL divergence tells me how much information do I need to distinguish
the weighted coin from the fair coin.
