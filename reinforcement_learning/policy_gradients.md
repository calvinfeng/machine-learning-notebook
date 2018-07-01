
# Policy Gradients
The problem with Q-learning is that Q-function can be very complicated. For a problem with high-dimensional state, it is hard to learn exact or accurate Q-value for every pair of state-action. However, policy can be much simpler. The question is can we learn a policy directly?


We will define a class of parametrized policies

$$
\prod = \left\{ \pi^{\theta}, \theta \in \mathbb{R}^{m} \right\}
$$

For each policy, we will define its value as

$$
J(\theta) = \mathbb{E} \left[ \Sigma \; \gamma^{t} r_{t} \mid \pi^{\theta} \right]
$$

We want to find the optimal policy that will give us the best expected reward.

$$
\theta^{*} = argmax_{\theta} \; J(\theta)
$$



## Reinforce Algorithm
### Formulation
Mathematically we can write

$$
J(\theta) = \int r(\tau)p(\tau;\theta)\; d\tau
$$ 

And $$r(\tau)$$ is the reward of a state transition trajectory

$$
\tau = (s_{0}, a_{0}, r_{0}, s_{1}, ...)
$$

We want to do gradient ascent to **maximize** the expected reward from the policy. So we need to
differentiate the integral!

$$
\nabla_{\theta} J(\theta) = \int r(\tau) \nabla_{\theta}\; p(\tau;\theta)\; d\tau
$$

However, this is intractable. Gradient of an expectation value is problematic because probability
depends on $$\theta$$. But, personally I don't see why this is the case. **NOTE**: Figure this shit out.

Here's a trick to do
$$
\nabla_{\theta} p(\tau;\theta) = p(\tau;\theta) \frac{\nabla_{\theta}p(\tau;\theta)}{p(\tau;\theta)} =
p(\tau;\theta)\nabla_{\theta}log\;p(\tau;\theta)
$$

Then we inject it back into the original integral
$$
\nabla_{\theta} J(\theta) = \int \left(r(\tau) \nabla_{\theta}\; log\; p(\tau;\theta)\right)p(\tau;\theta)\;d\tau
= \mathbb{E}\left[ r(\tau) \nabla_{\theta} log \; p (\tau;\theta) \right]
$$

We can estimate with Monte Carlo sampling.

### Gradient Estimation
How can we compute the integral without knowing the transition probabilities? We know that
probability of a state transition trajectory is the following.

$$
p(\tau\;;\theta) = \prod p(s_{t + 1} \mid s_{t}, a_{t}) \pi_{\theta}(a_{t} \mid s_{t})
$$

Now if we take the log of the above expression, we get the following.

$$
log\; p(\tau\;;\theta) = \Sigma\; log\; p(s_{t + 1} \mid s_{t}, a_{t}) + log\; \pi_{\theta}(a_{t} \mid s_{t})
$$

Once we differentiate this expression with respect to $\theta$ then we can see that it does not depend on transition probabilities!!!

$$
\nabla_{\theta}\; log\; p(\tau;\theta) = \Sigma\;\nabla_{\theta} log \pi_{\theta}(a_{t} \mid s_{t})
$$

Therefore, when we sample a trajectory $\tau$, we can estimate $J(\theta)$ with the following.

$$
\nabla_{\theta} J(\theta) = \Sigma_{t \geq 0} \; r(\tau) \nabla_{\theta} log\; \pi_{\theta}(a_{t} \mid s_{t})
$$

### Intuition
Now we have defined out gradient estimator.

$$
\nabla_{\theta} J(\theta) \approx \Sigma_{t \geq 0} \; r(\tau) \nabla_{\theta} log\; \pi_{\theta}(a_{t} \mid s_{t})
$$

Here is the interpretation:
* If reward from the trajectory $$\tau$$ is high, i.e. $$r(\tau)$$ is high, then gradient ascent will increase the probabilities of the actions seen.
* If reward from the  trajectory $$\tau$$ is low, i.e. $$r(\tau)$$ is low, then gradient ascent will decrease the probabilities of the actions seen.

It may seem simplistic to say that if a trajectory is good, then all its actions are good. Howevr, it averages out in expectation.

### Variance Reduction
Suppose you want to train the agent such that it always takes the best action in every time step for
a given state. The estimator does not specifically do that for you. It only looks at a whole
trajectory and makes some estimation about what is good and what is bad. It does not train itself to
make the best decision at every time step. Thus, although we may have a good reward trajectory,
individual actions within this trajectory are not guaranteed to be the best choice. However, it
works out if we have enough samples. The estimator requires a lot of samples to become unbiased in
its gradient estimation. The challenge is, how can we reduce variance when samples are small.

#### Approach #1
Push up probabilities of an action seen, only by the cumulative future reward from that state.

$$
\nabla_{\theta}J(\theta) \approx \Sigma_{t \geq 0} \; \left( \Sigma_{t^{\prime} \geq t} r_{t^{\prime}} \right) \nabla_{\theta} log\; \pi_{\theta} (a_{t} \mid s_{t})
$$

#### Approach #2
Use discount factor gamma to ignore delayed effects

$$
\nabla_{\theta}J(\theta) \approx \Sigma_{t \geq 0} \; \left( \Sigma_{t^{\prime} \geq t} \gamma^{t^{\prime} - t} r_{t^{\prime}} \right) \nabla_{\theta} log\; \pi_{\theta} (a_{t} \mid s_{t})$$

## AlphaGo Example
AlphaGo employs a mix of supervised learning and reinforcement learning to beat world champions in Go. Here is the high level overview of how AlphaGo was trained and implemented.

* Featurize the board by stone color and positions, move legality, bias and etc...
* Initialize a policy network with supervised learning from professional Go games, i.e. using a deep neural network to map states to actions. 
* Continue the training using policy gradient by playing against itself from random previous iterations, +1 or -1 reward for winning or losing.
* Learn value network for critic
* Finally, combine policy and value networks in a Monte Carlo Tree Search algorithm to select actions by look ahead search.
