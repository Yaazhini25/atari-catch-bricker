"""
Watch the trained Q-learning agent play Brick Catcher, rendered with Pygame.

Usage:
    python play.py            # loads q_table.pkl and plays greedily
    python play.py --train    # trains live on-screen (slower, but visual)
"""

import sys
import pygame
from brick_env import BrickEnv
from q_agent import QLearningAgent

CELL = 50
GRID_W, GRID_H = 10, 10
SCREEN_W, SCREEN_H = GRID_W * CELL, GRID_H * CELL
FPS = 12

WHITE = (245, 245, 245)
BLACK = (20, 20, 20)
RED = (220, 60, 60)
BLUE = (60, 120, 220)
GRAY = (200, 200, 200)


def draw(screen, env, font, score, episodes, epsilon=None):
    screen.fill(WHITE)

    # grid lines
    for x in range(0, SCREEN_W, CELL):
        pygame.draw.line(screen, GRAY, (x, 0), (x, SCREEN_H))
    for y in range(0, SCREEN_H, CELL):
        pygame.draw.line(screen, GRAY, (0, y), (SCREEN_W, y))

    # brick
    bx, by = env.brick_x, env.brick_y
    pygame.draw.rect(screen, RED, (bx * CELL + 8, by * CELL + 8, CELL - 16, CELL - 16))

    # paddle (drawn at the bottom row, spanning a wider hit zone)
    px = env.paddle_x
    pygame.draw.rect(screen, BLUE, (px * CELL, SCREEN_H - CELL, CELL, CELL // 3))

    # HUD
    label = f"Score: {score}   Episodes: {episodes}"
    if epsilon is not None:
        label += f"   epsilon: {epsilon:.3f}"
    text = font.render(label, True, BLACK)
    screen.blit(text, (10, 10))

    pygame.display.flip()


def main():
    live_train = "--train" in sys.argv

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Q-Learning Brick Catcher")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    env = BrickEnv(GRID_W, GRID_H)
    agent = QLearningAgent(grid_w=GRID_W, grid_h=GRID_H)

    if not live_train:
        try:
            agent.load("q_table.pkl")
            agent.epsilon = 0  # greedy / no exploration when just demoing
            print("Loaded trained Q-table. Playing greedily.")
        except FileNotFoundError:
            print("No q_table.pkl found — run `python train.py` first, "
                  "or run `python play.py --train` to train live.")
            return

    state = env.reset()
    score = 0
    episodes = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = agent.choose_action(state, greedy=not live_train)
        next_state, reward, done = env.step(action)

        if live_train:
            agent.update(state, action, reward, next_state, done)

        state = next_state

        if done:
            if reward > 0:
                score += 1
            episodes += 1
            if live_train:
                agent.decay_epsilon()
            state = env.reset()

        draw(screen, env, font, score, episodes,
             epsilon=agent.epsilon if live_train else None)
        clock.tick(FPS if not live_train else 60)

    if live_train:
        agent.save("q_table.pkl")
        print("Saved live-trained Q-table to q_table.pkl")

    pygame.quit()


if __name__ == "__main__":
    main()
