"""
Q-Learning Agent for Brick Catcher Game
-----------------------------------------
A simple tabular Q-learning agent using NumPy.

State  = (paddle_x, brick_x, brick_y)   -> discretized grid positions
Action = 0 (move left), 1 (stay), 2 (move right)
"""

import numpy as np
import random
import pickle


class QLearningAgent:
    def __init__(self, grid_w, grid_h, n_actions=3,
                 alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.n_actions = n_actions

        self.alpha = alpha          # learning rate
        self.gamma = gamma          # discount factor
        self.epsilon = epsilon      # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table shape: [paddle_x, brick_x, brick_y, action]
        self.q_table = np.zeros((grid_w, grid_w, grid_h, n_actions))

    def choose_action(self, state, greedy=False):
        """Epsilon-greedy action selection."""
        if (not greedy) and (random.random() < self.epsilon):
            return random.randint(0, self.n_actions - 1)
        p, b_x, b_y = state
        return int(np.argmax(self.q_table[p, b_x, b_y]))

    def update(self, state, action, reward, next_state, done):
        """Bellman equation Q-value update."""
        p, b_x, b_y = state
        np_, nb_x, nb_y = next_state

        current_q = self.q_table[p, b_x, b_y, action]
        max_future_q = 0 if done else np.max(self.q_table[np_, nb_x, nb_y])

        target = reward + self.gamma * max_future_q
        self.q_table[p, b_x, b_y, action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
