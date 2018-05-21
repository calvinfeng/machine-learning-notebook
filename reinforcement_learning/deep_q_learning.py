# Author(s) Calvin Feng

import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque


class DeepQAgent(object):
    def __init__(self, state_dim, action_dim):
        """Initialize a deep Q agent.

        Args:
            state_dim (int): Dimension of state, e.g. cart position, cart velocity, and etc...
            action_dim (int): Dimension of executable actions, in case this, left & right.

        Properties:
            state_dim (int): Dimension of state, e.g. cart position, cart velocity, and etc...
            action_dim (int): Dimension of executable actions, in case this, left & right.
            gamma (float): Future reward discount rate.
            epsilon (float): Probability for choosing random policy.
            epsilon_decay (float): Rate at which epsilon decays toward zero.
            learning_rate (float): Learning rate for Adam optimizer.
            model (keras.Sequential): Sequential neural network model.

        Returns:
            agent (DeepQAgent)            
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 5e-3
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_dim, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Append a new memory tuple state to the memory deque
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Take action on a given state
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        policies = self.model.predict(state)
        return np.argmax(policies[0])

    def replay(self, batch_size):
        """Replay the experience and perform model update on every experience replay.
        """
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_action_values = self.model.predict(state)
            target_action_values[0][action] = target

            self.model.fit(state, target_action_values, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay



if __name__ == '__main__':
    # Create the Cartpole game
    env = gym.make('CartPole-v0')
    env._max_episode_steps = None
    
    # Extract environment dimensions
    action_dim = env.action_space.n
    state_dim, = env.observation_space.shape

    agent = DeepQAgent(state_dim=state_dim, action_dim=action_dim)
    max_episodes = 5000
    batch_size = 32


    # state[0]: cart position range from -2.4 to 2.4
    # state[1]: cart velocity range from -Inf to Inf
    # state[2]: pole angle range from -41.8 to 41.8
    # state[3]: pole velocity at tip range from -Inf to Inf
    for e in range(max_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_dim])
        score = 0
        game_over = False
        while not game_over:
            env.render()
            action = agent.act(state)

            # Reward is +1 for every action taken
            next_state, reward, game_over, _ = env.step(action)
            
            # Penalize the action if the action leads to a game over
            if game_over:
                reward = -10
            
            score += reward
            
            # Record the experience
            next_state = np.reshape(next_state, [1, state_dim])
            agent.remember(state, action, reward, next_state, game_over)
            state = next_state

            if game_over:
                print 'Epsiode: {}/{} is over! Score: {}, epsilon: {:.2}'.format(e+1, 
                                                                                 max_episodes, 
                                                                                 score, 
                                                                                 agent.epsilon)
            
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
