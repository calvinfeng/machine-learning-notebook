# Approximate Inference

> Many probablistic models are difficult to train because it is difficult to perform inference in them. In the context of deep learning, we usually have a set of visible variables $x$ and a set of latent variables $z$. The challenge of inference refers to the difficult problem of computing posterior $p(z \mid x)$ or taking expectations with respect to it. Such operations are often necessary for tasks like maximum likelihood learning.

In deep learning, the posterior in general means given a visible variable, e.g. an input, what's the probability distribution of the latent variables e.g. hidden layer activation. So we have posterior of the following form.

$$
p(z\mid x) = \frac{p(x \mid z) p(z)}{p(x)} = \frac{p_{joint}(z, x)}{p(x)}
$$

However, the problem is we can't compute the denominator a.k.a. the marginal probability `p(x)`.

$$
p(x) = \int_{-\infty}^{\infty} p(z, x) dz
$$

This integral requires us to sum over all possible values of `z`. There is no closed form solution of this integral over a joint distribution. We have to iterate through all possible values of `z`. It becomes unfeasible if `z` is high dimensional vector. Thus, we need a way to approximate this posterior $p(z /mid x)$.

## Inference as Optimization

> Many approaches to confronting the problem of difficult inference make use of the observation that exact inference can be described as an optimization problem.

### ELBO

Resources
- [Variational Inference, Part 1](https://www.youtube.com/watch?v=UTMpM4orS30)
- [Variational Inference, Part 2](https://www.youtube.com/watch?v=VWb0ZywWpqc)
- [Variational Inference, Part 3](https://www.youtube.com/watch?v=4LuA5m5Hsxc)
- [Deep Learning Book](https://www.deeplearningbook.org/contents/inference.html)

Assume that we have a probabilistic model with parameters $\theta$. It takes observed input $x$ and generates latent output $z$. We want to update $\theta$ such that it maximizes the likelihood of our model producing the observed data distribution. We will never know the true distribution of our inputs because we are only given the snapshots. For example, if `x`s are pixels of images, it's impossible for us to know the true distribution of pixels across all images. At best, we can only come up with a modeled distribution that aims to maximizes the likelihood of observed data.

$$
\text{argmax}_\theta \Sigma_{i=1}^{N} log\;p_\theta(x_i)
$$

It is too difficult to calculate the distribution of $x$ because we need to marginalize out $z$. We can get away by computing a lower bound on $log\;p_\theta(x)$. This bound is called the **evidence lower bound**(ELBO). 

$$
\mathbb{L}(x, \theta, q) = log\;p_\theta(x) - D_{KL}\left[\; q(z \mid x) \;\|\; p_\theta(z \mid x) \right] \text{where q is an arbitrary probability distribution over z}
$$


The difference between the $log\;p_\theta(x)$ and $\mathbb{L}$ is the KL divergence term. KL divergence is always non-negative. We can see that lower bound becomes the true distribution if we can minimize the KL divergence term to 0. It goes to zero when $q$ is the same distribution as $p(z \mid x)$.

We can re-arrange $\mathbb{L}$ algebraically.

$$
\begin{align}
\mathbb{L}(x, \theta, q) 
&= log\;p_\theta(x) - D_{KL}\left[\; q(z \mid x) \;\|\; p_\theta(z \mid x) \right] \\ 
&= log\;p_\theta(x) - \mathbb{E}_{z\sim q} log\frac{q(z \mid x)}{p(z \mid x)} \\
&= log\;p_\theta(x) - \mathbb{E}_{z\sim q} log\left( q(z\mid x) \frac{p_\theta(x)}{ p_\theta(x, z)} \right) \\
&= log\;p_\theta(x) - \mathbb{E}_{z\sim q} \left[ log\;q(z \mid x) - log\;p_\theta(x, z) + log\;p_\theta(x) \right]
\end{align}
$$

Since we have expected value of $log\;p_\theta(x)$ with respect to $h$, we can say this is taking the expected value of a constant. This will cancel out the first term. Here we have the final form.

$$
\mathbb{L}(x, \theta, q) = - \mathbb{E}_{z\sim q} \left[ log\;q(z \mid x) - log\;p_\theta(x, z)\right]
$$

The inference can be thought of as the procedure for finding the $q$ that maximizes the lower bound $\mathbb{L}$. Whether the lower bound is tight (close approximation to $p(x)$) or loose, it's dependent on the choice of $q$ we pick. $\mathbb{L}$ is significantly easier to compute when we choose easy distribution $q$, e.g. a Gaussian distribution with mean and variance as the only parameters. 

## TensorFlow Example

This example is taken from [Convolutional Variational Autoencoder](https://www.tensorflow.org/tutorials/generative/cvae).

Given that we have the ELBO definition from section above. We can define a lower bound for our loss as 

$$
\begin{align}
L
&= log\;p_\theta(x) \geq \mathbb{E}_{z\sim q}\left[ log\;p_\theta(x, z) - log\;q(z \mid x) \right] \\
&\geq \mathbb{E}_{z\sim q}\left[ log\;p_\theta(x \mid z) + log\;p_\theta(z) - log\;q(z \mid x) \right]
\end{align}
$$

This is because joint distribution can be written as conditional probability.

$$
p_\theta(x, z) = p_\theta(x \mid z) p_\theta(z)
$$

In practice, we cannot compute the exact expected value of the expression. We will rely on a single sample per forward propagation to perform a Monte Carlo estimate.

$$
L = log\;p(x \mid z) + log\;p(z) - log\;q(z\mid x)
$$

This is quite similar to the idea in cross entropy where we try to learn the distribution of labels by measuring the difference target and predicted distribution. We cannot know the exact expected value of target distribution so we approximate it using the same technique.


```python
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, Sequential


class CVAE(tf.keras.Model):
    """Convolutional Variational Autoencoder."""
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            layers.Flatten(),
            # No activation
            layers.Dense(latent_dim + latent_dim),
        ])

        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            layers.Dense(units=7*7*32, activation=tf.nn.relu),
            layers.Reshape(target_shape=(7, 7, 32)),
            layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            # No activation
            layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
        ])

    @tf.function
    def sample(self, z=None):
        if z is None:
            z = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(z, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
```

The VAE has encoder and decoder. The encoder consumes an observable variable `x` vector and produces a latent variable `z` via reparametrization. We choose Gaussian distribution to be our $q(z \mid x)$ distribution.

**Why Reparametrization?**

`z` is supposed to be sampled from a Gaussian distribution but gradient cannot flow through a `tf.random.normal` function. We need to reparametrize `z` such that the gradient is not dependent on `tf.random.normal`.

We can generate a unit Gaussian from `tf.random.normal` and redefine `z` as follows.

$$
z = \mu + \sigma \cdot \epsilon
$$

where $\epsilon$ is sampled from a unit Gaussian distribution.


```python
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log_2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log_2pi), axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    log_p_x_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    log_p_z = log_normal_pdf(z, 0., 0.)
    log_q_z_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(log_p_x_z + log_p_z - log_q_z_x)
```

This example uses `logvar` for numerical stability. We can get rid of the `log` from `tf.exp(logvar / 2)` because 

$$
log(\text{variance}) = log(\sigma^2) = 2\;log(\sigma)
$$

$$
\sigma = e^{(log(\text{variance}) / 2)}
$$


```python
latent_dim = 10
model = CVAE(latent_dim)
x = np.random.randn(1, 28, 28, 1)
print('Shape of input x', x.shape)
mean, logvar = model.encode(x)
z = model.reparameterize(mean, logvar)
print('Shape of latent vector z', z.shape)
y = model.decode(z)
print('Shape of output y', y.shape)

print(tf.reduce_sum(z))
tf.exp(log_normal_pdf(z, 0., 0.))
```

    Shape of input x (1, 28, 28, 1)
    Shape of latent vector z (1, 10)
    Shape of output y (1, 28, 28, 1)
    tf.Tensor(-0.26499367, shape=(), dtype=float32)





    <tf.Tensor: shape=(1,), dtype=float32, numpy=array([2.3094783e-06], dtype=float32)>



## Loss Explained

Let $x$ be our input and $\hat{x}$ be our output. Our objective is to set $x \approx \hat{x}$

```py
logp_x_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
```

$log\;p(x\mid z)$ is equivalent to asking how far apart is the distribution of output $\hat{x}$ given that we have `z` (which comes from input $x$) away from the distribution of $x$. This is known as the **reconstruction loss**.

```py
log_p_z = log_normal_pdf(z, 0., 0.)
log_q_z_x = log_normal_pdf(z, mean, logvar)
```

These two terms reprsent the **KL divergence**. It asks how far apart is model's encoder output `z` distribution away from the expected Gaussian distribution of `z`. This divergence is expressed in terms of log density ratio which derivation can be found on [Density Ratio Estimation for KL Divergence Minimzation Between Implicit Distributions](https://tiao.io/post/density-ratio-estimation-for-kl-divergence-minimization-between-implicit-distributions/). 

When we set `mean=0` and `logvar=0` for $p(z) = $ `log_normal_pdf(z, 0, 0)`. We will obtain `mean=0` and `var=1` which is saying that `z` is sampled from a standard Gaussian probability density function.

If the loss is minimized, model $p(z)$ will match the enforced Gaussian distribution $q(z \mid x)$. We selectively chose a distribution for $q$. This loss minimization is encouraging the model to learn the selected distribution $q$.


Another way to write the KL divergence loss without the log normal probability density function is

```py
kl_loss = -0.5 * (1 + logvar - mean**2 - tf.exp(logvar)
```
