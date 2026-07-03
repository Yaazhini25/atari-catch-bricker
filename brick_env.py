"""
Brick Catcher Environment
--------------------------
A simple grid-based "catch the falling brick" game environment,
independent of rendering, so it can be trained fast (headless)
or watched live with Pygame.
"""

import random

GRID_W = 10   # number of horizontal cells (paddle can be at 0..GRID_W-1)
GRID_H = 10   # number of vertical cells the brick falls through


class BrickEnv:
    def __init__(self, grid_w=GRID_W, grid_h=GRID_H):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.reset()

    def reset(self):
        self.paddle_x = self.grid_w // 2
        self.brick_x = random.randint(0, self.grid_w - 1)
        self.brick_y = 0
        return self._get_state()

    def _get_state(self):
        return (self.paddle_x, self.brick_x, self.brick_y)

    def step(self, action):
        """
        action: 0 = left, 1 = stay, 2 = right
        returns: next_state, reward, done
        """
        if action == 0:
            self.paddle_x = max(0, self.paddle_x - 1)
        elif action == 2:
            self.paddle_x = min(self.grid_w - 1, self.paddle_x + 1)
        # action == 1 -> stay

        self.brick_y += 1
        done = False
        reward = -0.01  # small penalty per step to encourage efficient catching

        if self.brick_y >= self.grid_h - 1:
            done = True
            if self.paddle_x == self.brick_x:
                reward = 1.0     # caught the brick!
            else:
                reward = -1.0    # missed it

        return self._get_state(), reward, done
