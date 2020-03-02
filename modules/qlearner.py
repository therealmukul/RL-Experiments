"""Q-Learner class."""

# System Library
import random

# Third-party
import numpy as np


###############################################################################
class QLearner:
    """Class for performing Q-Learning on OpenAI Gym Environments."""

    def __init__(self, env, alpha, gamma, epsilon, name=None, verbose=False):
        """Initializations"""
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = name if name else ''
        self.verbose = verbose

        self.num_actions = env.action_space.n
        self.num_states = env.observation_space.n
        self.Q = np.zeros((self.num_states, self.num_actions))

        self.episode_rewards = []

        if self.verbose:
            if self.name:
                print(f'Q-Learner-{self.name}')
            else:
                print('Q-Learner')
            print(f'\talpha   = {self.alpha}')
            print(f'\tgamma   = {self.gamma}')
            print(f'\tepsilon = {self.epsilon}')
            print(f'\tactions = {self.num_actions}')
            print(f'\tstates  = {self.num_states}')
            print(f'\tQ dims  = {self.Q.shape}')

    def choose_action(self, state):
        """Epsilon greedy exploration vs exploitation strategy."""
        rand = random.uniform(0, 1)

        if rand > self.epsilon:
            action = np.argmax(self.Q[state, :])
        else:
            action = self.env.action_space.sample()

        return action

    def learn(self, s, a, r, s_):
        """Update Q table.

        Q[s, a] = Q[s, a] + alpha * (r[s, a] + gamma * max(Q[s_, a_]) - Q[s, a])
        """

        self.Q[s, a] += self.alpha * \
                        (r + self.gamma * np.max(self.Q[s_, :]) - self.Q[s, a])

    def train(self, n_episodes, print_interval=None):
        """Train agent."""
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_rewards = 0

            while not done:
                action = self.choose_action(state)
                state_, reward, done, info = self.env.step(action)

                self.learn(state, action, reward, state_)

                state = state_
                episode_rewards += reward

            self.episode_rewards.append(episode_rewards)

            if self.verbose:
                if print_interval:
                    if episode % print_interval == 0:
                        print(f'episode {episode}, reward = {episode_rewards}')
