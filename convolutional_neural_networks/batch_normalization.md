# Batch Normalization

## Unit Gaussian Activations

Batch normalization is *invented* and widely popularized by the paper *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*.
In deep neural network, activations between neural layers are extremely dependent on the parameter
initialization, which in turn affects how outputs are backprop into each layer during training. Poor
initialization can greatly affect how well a network is trained and how fast it can be trained.
Networks train best when each layer has an unit Gaussian distribution for its activations. So if you
really want unit Gaussian activations, you can make them so by applying batch normalization to every
layer.

Basically, batch normalization is a powerful technique for decoupling the weight updates from
parameter initialization. Quoted from the paper, *batch normalization allows us to use much higher learning rates and be less careful about initialization.* Let's consider a batch of activations at some layer, we can make each dimension (denoted by $$k$$)
unit Gaussian by applying:

$$
\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{Var[x^{(k)}]}}
$$

Each batch of training example has dimension `D`. Compute the empirical mean and variance
independently for each dimension by using all the training data. Batch normalization is usually
inserted after fully connected or convolutional layers and before nonlinearity is applied. For the
convolutional layer, we are basically going to have one mean and one standard deviation per
activation map that we have. And then we are going to normalize across all of the examples in the
batch of data.

## Avoid Constraints by Learning

If we have a `tanh` layer, we don't really want to constraint it to the linear regime. The act of
normalization might force us to stay within the center, which is known as the linear regime. We want
flexibility so ideally we should learn batch normalization as a paramter of the network. In other
words, we should insert a parameter which can be learned to effectively cancel out batch
normalization if the network sees fit.

We will apply the following operation to each normalized vector:

$$
y^{(k)} = \gamma^{(k)}\hat{x}^{(k)} + \beta^{(k)}
$$

Such that the network can learn

$$
\gamma^{(k)} = \sqrt{Var[x^{(k)}]}
$$
$$
\beta^{(k)} = E[x^{(k)}]
$$

And effectively recover the identity mapping as if you didn't have batch normalization, i.e. to
cancel out the batch normalization if the network sees fit.

## Procedure

**Inputs**: Values of $$x$$ over a mini-batch: **B** = $$\{x_{1}...x_{m}\}$$

**Outputs**: $$\{y_{i} = BN_{\gamma, \beta}(x_{i})\}$$

Find mini-batch mean:
$$
\mu_{B} = \frac{1}{m} \sum^{m}_{i = 1} x_{i}
$$

Find mini-batch variance:
$$
\sigma_{B}^{2} = \frac{1}{m} \sum^{m}_{i = 1} (x_{i} - \mu_{B})^{2}
$$

Normalize:
$$
\hat{x_{i}} = \frac{x_{i} - \mu_{B}}{\sqrt{\sigma_{B}^{2} + \epsilon}}
$$

Scale and shift:
$$
y_{i} = \gamma \hat{x_{i}} + \beta = BN_{\gamma, \beta}(x_{i})
$$

### Benefits

* Improves gradient flow through the network
* Allows higher learning rates
* Reduces the strong dependence on initialization
* Acts as a form of regularization in a funny way, and slightly reduces the need for dropout

## Detailed Implementation & Derivation

Here comes the derivation; much of the derivation comes from the paper itself and also from Kevin
Zakka's blog on Github.

### Notations

* **BN** stands for batch normalization
* $$x$$ is the input matrix/vector to the **BN** layer
* $$\mu$$ is the batch mean
* $$\sigma^{2}$$ is the batch variance
* $$\epsilon$$ is a small constant added to avoid dividing by zero
* $$\hat{x}$$ is the normalized input matrix/vector
* $$y$$ is the linear transformation which scales $$x$$ by $$\gamma$$ and $$\beta$$
* $$f$$ represents the next layer after **BN** layer, if we assume a forward pass ordering

### Forward Pass

Forward pass is very easy intuitively and mathematically.

First we find the mean across a mini-batch of training examples

$$
\mu = \frac{1}{m} \sum^{m}_{i = 1} x_{i}
$$

Find the variance across the same mini-batch of training examples

$$
\sigma^{2} = \frac{1}{m} \sum^{m}_{i = 1} (x_{i} - \mu)^{2}
$$

And then apply normalization

$$
\hat{x_{i}} = \frac{x_{i} - \mu}{\sqrt{\sigma^{2} + \epsilon}}
$$

Finally, apply linear transformation with learned parameters to enable network to recover identity.
In case we wonder why do we need to do this.

> Note that simply normalizing each input of a layer may change what the layer can represent. For
> instance, normalizing the inputs of a sigmoid would constrain them to the linear regime of the
> nonlinearity. To address this, w make sure that the transformation inserted in the network can
> represent the identity transform.

$$
y_{i} = \gamma \hat{x_{i}} + \beta = BN_{\gamma, \beta}(x_{i})
$$

If $$\gamma$$ is 1 and $$beta$$ is 0 then the linear transformation is an identity transformation.

```python
import numpy as np

def batch_norm_forward(x, gamma, beta, bn_params):
    eps = bn_params.get('eps', 1e-5)
    momentum = bn_params.get('momentum', 0.9)
    mode = bn_params.get('mode', 'train')

    N, D = x.shape
    running_mean = bn_params.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_params.get('running_var', np.zeros(D, dtype=x.dtype))

    y = None
    if mode == 'train':
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        x_norm = (x - mean) / np.sqrt(var + eps)
        y = x_norm * gamma + beta

        # Update running mean and running variance during training time
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

    elif mode == 'test':
        # Use running mean and runningvariance for making test predictions
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        y = x_norm * gamma + beta
    else:
        raise ValueError('Invalid forward pass batch norm mode %s' % mode)

    bn_params['running_mean'] = running_mean
    bn_params['running_var'] = running_var

    return y

x = np.random.rand(4, 4)
bn_params = {}
y = batch_norm_forward(x, 1, 0, bn_params)
```

```python
print y.mean(axis=0, keepdims=True)
```

    [[ -2.22044605e-16  -5.55111512e-16   0.00000000e+00   0.00000000e+00]]

As we can see that the output has a mean centered at zero.

```python
print y.var(axis=0, keepdims=True)
```

    [[ 0.99973824  0.99895175  0.99928567  0.9996811 ]]

And variance of one across all examples.

```python
print bn_params['running_var']
print bn_params['running_mean']
```

    [ 0.00381932  0.00095297  0.00139892  0.00313479]
    [ 0.04364179  0.06298149  0.03975495  0.03415751]

### Backward Pass & Gradient Computations

Now here comes the hard part. We are given an upstream gradient, i.e. the gradient of loss function
w.r.t to output of the batch normalization layer.

$$
\frac{\partial L}{\partial y} = \frac{\partial L}{\partial f} \frac{\partial f}{\partial y}
$$

We need to find

$$
\frac{\partial L}{\partial \hat{x}}, \; \frac{\partial L}{\partial \sigma^{2}}, \; \frac{\partial L}{\partial \mu}, \; \frac{\partial L}{\partial x}, \; \frac{\partial L}{\partial \gamma}, \; and \; \frac{\partial L}{\partial \beta}
$$

#### Gradient of Normalized Input

The derivative of $y$ with respect to $\hat{x}$ is simple:

$$
\frac{\partial y}{\partial \hat{x}} = \gamma
$$

Thus,

$$
\frac{\partial L}{\partial \hat{x}} = \frac{\partial L}{\partial y} \gamma
$$

In Python,

```python
grad_x_norm = grad_y * gamma # Element-wise multiplication
```

#### Gradient of Gamma

The derivative of $$y$$ with respect to $$\gamma$$ is:

$$
\frac{\partial y}{\partial \gamma} = \hat{x}
$$

Thus,

$$
\frac{\partial L}{\partial \gamma} = \sum^{m}_{i=1}\frac{\partial L}{\partial y_{i}} \cdot \hat{x}_{i}
$$

We need to perform a sum across all training examples in the mini-batch and squash the shape
`(N, M)` to `(M,)`

In Python,

```python
grad_gamma = (grad_y * x_norm).sum(axis=0)
```

#### Gradient of Beta

The derivative of $$y$$ with respect to $$\beta$$ is:

$$
\frac{\partial y}{\partial \beta} = 1
$$

Thus,

$$
\frac{\partial L}{\partial \beta} = \sum^{m}_{i=1}\frac{\partial L}{\partial y_{i}}
$$

We need to perform a sum across all training examples in the mini-batch and squash the shape
`(N, M)` to `(M,)`

In Python,

```python
grad_beta = grad_y.sum(axis=0)
```

#### Gradient of Variance

The derivative of $$\hat{x}$$ with respect to $$\sigma^{2}$$ is:

$$
\frac{\partial \hat{x}}{\partial \sigma^{2}} = \frac{-1}{2} (x - \mu) (\sigma^{2} + \epsilon)^{-3/2}
$$

Thus,

$$
\frac{\partial L}{\partial \sigma^{2}} = \sum^{m}_{i=1} \frac{\partial L}{\partial \hat{x}_{i}} (\frac{-1}{2}) (x_{i} - \mu) (\sigma^{2} + \epsilon)^{-3/2}
$$

We need to perform a sum across all training examples in the mini-batch and squash the shape
`(N, M)` to `(M,)`

In Python,

```python
dvar = (-0.5) * (x - mean) * (var + eps)**(-3.0/2)
grad_var = np.sum(grad_x_norm * dvar, axis=0)
```

#### Gradient of Mean

We are going to use chain rule to solve for this gradient:

$$
\frac{\partial L}{\partial \mu} = \frac{\partial L}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial \mu} + \frac{\partial L}{\partial \sigma^{2}} \cdot \frac{\partial \sigma^{2}}{\partial \mu}
$$

$$
\frac{\partial \hat{x}}{\partial \mu}  = \frac{-1}{\sqrt{\sigma^{2} + \epsilon}}
$$

$$
\frac{\partial \sigma^{2}}{\partial \mu} = \frac{-2}{m}\sum_{i=1}^{m} (x_{i} - \mu)
$$

Thus,

$$
\frac{\partial L}{\partial \mu} =  \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_{i}} \frac{-1}{\sqrt{\sigma^{2} + \epsilon}} + \frac{\partial L}{\partial \sigma^{2}} \frac{-2}{m}\sum_{i=1}^{m} (x_{i} - \mu)
$$

In Python,

```python
dxnorm_dmean = -1 / np.sqrt(var + eps)
dvar_dmean = np.sum((-2 / x.shape[0]) * (x - mean), axis=0)
grad_mean = np.sum(grad_x_norm * dxnorm_dmean, axis=0) + grad_var * dvar_dmean
```

#### Gradient of Input

Use chain rule again to solve for the final gradient:

$$
\frac{\partial L}{\partial x} =  \frac{\partial L}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial x} + \frac{\partial L}{\partial \sigma^{2}} \cdot \frac{\partial \sigma^{2}}{\partial x} + \frac{\partial L}{\partial \mu} \cdot \frac{\partial \mu}{\partial x}
$$

Now fill in the missing pieces:

$$
\frac{\partial \hat{x}}{\partial x} = \frac{1}{\sqrt{\sigma^{2} + \epsilon}}
$$

$$
\frac{\partial \sigma^{2}}{\partial x} = \frac{2 (x - \mu)}{m}
$$

$$
\frac{\partial \mu}{\partial x} = \frac{1}{m}
$$

Now we just plug and chuck

$$
\frac{\partial L}{\partial x_{i}} =  \frac{\partial L}{\partial \hat{x}_{i}} \cdot \frac{1}{\sqrt{\sigma^{2} + \epsilon}} + \frac{\partial L}{\partial \sigma^{2}} \cdot \frac{2 (x_{i} - \mu)}{m} + \frac{\partial L}{\partial \mu} \cdot \frac{1}{m}
$$

In Python,

```python
dxnorm_dx = 1 / np.sqrt(var + eps)
dvar_dx = 2 * (x - mean)
dmean_dx = 1 / x.shape[0]
grad_x = grad_x_norm * dxnorm_dx + grad_var * dvar_dx + grad_mean * dmean_dx
```

## Simplification

Work on this later...
