"""
Train the Q-learning agent on the Brick Catcher environment.
Runs headless (no Pygame window) for speed, then saves the Q-table.
"""

import numpy as np
from brick_env import BrickEnv
from q_agent import QLearningAgent

EPISODES = 20000

def train():
    env = BrickEnv()
    agent = QLearningAgent(grid_w=env.grid_w, grid_h=env.grid_h)

    rewards_history = []
    catch_count = 0

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if total_reward > 0:
            catch_count += 1

        agent.decay_epsilon()
        rewards_history.append(total_reward)

        if ep % 1000 == 0:
            recent_catch_rate = sum(1 for r in rewards_history[-1000:] if r > 0) / 10.0
            print(f"Episode {ep:6d} | epsilon={agent.epsilon:.3f} | "
                  f"catch rate (last 1000)={recent_catch_rate:.1f}%")

    agent.save("q_table.pkl")
    print(f"\nTraining complete. Overall catch rate: {catch_count / EPISODES * 100:.1f}%")
    print("Q-table saved to q_table.pkl")


if __name__ == "__main__":
    train()
