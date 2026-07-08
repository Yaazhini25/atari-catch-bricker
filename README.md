# Q-Learning Brick Catcher

A reinforcement learning agent that learns to play a brick-catching game
through pure reward optimization (tabular Q-learning), built with
**Python, Pygame, and NumPy**.

## How it works

- **Environment**: A paddle at the bottom of a 10x10 grid must catch a
  brick falling from the top.
- **State**: `(paddle_x, brick_x, brick_y)`
- **Actions**: move left, stay, move right
- **Reward**: `+1` for catching the brick, `-1` for missing it,
  `-0.01` per step (encourages efficient movement).
- **Algorithm**: Tabular Q-learning with epsilon-greedy exploration.
  The agent updates its Q-table using the Bellman equation after every
  step, and epsilon decays over time so it shifts from random
  exploration to exploiting what it has learned.
- Converges to ~100% catch rate after roughly 15-20k episodes.

## Project structure
```
brick_env.py   -> game environment (state, step, reward logic)
q_agent.py     -> QLearningAgent class (Q-table, action selection, updates)
train.py       -> headless training loop (fast, no rendering)
play.py        -> Pygame visual demo (watch the trained agent play)
web/index.html -> standalone browser version (for a public demo link)
q_table.pkl    -> a pretrained Q-table (already included, ~100% catch rate)
```

## Run locally

```bash
pip install pygame numpy

# train from scratch (takes ~10-20 seconds, saves q_table.pkl)
python train.py

# watch the trained agent play
python play.py

# OR watch it learn live, on-screen
python play.py --train
```

## Deploy the demo publicly (2 minutes, for your interview link)

Pygame can't run in a browser, so `web/index.html` is a lightweight
JS port of the exact same Q-learning algorithm — runs entirely
client-side, no server needed. Easiest ways to get a public link:

**Option A — GitHub Pages (recommended)**
1. Push this folder to a new GitHub repo.
2. Go to repo **Settings → Pages → Deploy from branch → main → /web** folder (or move `index.html` to repo root).
3. Your live link will be `https://<your-username>.github.io/<repo-name>/`

**Option B — Netlify Drop (fastest, no git needed)**
1. Go to https://app.netlify.com/drop
2. Drag the `web` folder onto the page.
3. Get an instant public URL.

Either way, open the link — the agent starts learning live in the
browser, and you can bump the speed slider to 25x to show it converge
in seconds during your interview.

## Talking points for your interview

- Why Q-learning (off-policy, model-free, simple tabular case fits a
  small discretized state space well).
- Exploration vs exploitation tradeoff via epsilon-greedy + decay.
- Reward shaping choice (small step penalty vs sparse-only reward).
- Convergence behavior / catch-rate curve.
- Limitations vs Deep Q-Networks (DQN) — tabular Q-learning doesn't
  scale to large/continuous state spaces (e.g. raw pixels), which is
  where a neural network Q-function (DQN) would be the natural next
  step — good to mention as a "future work" extension.
