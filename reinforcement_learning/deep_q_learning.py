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
            state_dim (int): Dimension of possible states
            action_dim (int): Dimension of possible actions            
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
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
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            print 'acting randomly!'
            return random.randrange(self.action_dim)
        
        policies = self.model.predict(state)
        return np.argmax(policies[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            target_policy = self.model.predict(state)
            target_policy[0][action] = target

            self.model.fit(state, target_policy, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


EPISODES = 5000


if __name__ == '__main__':
    # Create the Cartpole game
    env = gym.make('CartPole-v0')
    env._max_episode_steps = None
    
    # Extract environment dimensions
    action_dim = env.action_space.n
    state_dim, = env.observation_space.shape

    agent = DeepQAgent(state_dim=state_dim, action_dim=action_dim)
    batch_size = 32

    for e in range(EPISODES):
        # state[0]: cart position range from -2.4 to 2.4
        # state[1]: cart velocity range from -Inf to Inf
        # state[2]: pole angle range from -41.8 to 41.8
        # state[3]: pole velocity at tip range from -Inf to Inf
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
                                                                                 EPISODES, 
                                                                                 score, 
                                                                                 agent.epsilon)
            
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
