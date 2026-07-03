# Q-Learning Brick Catcher — Interview Prep Guide

## 1. The 30-second pitch

"I built a reinforcement learning agent that learns to play a brick-catching
game purely through trial and error. It's tabular Q-learning implemented in
Python and NumPy, with Pygame for visualization. The agent has no rules
programmed into it about how to play — it only receives a reward signal
(+1 for catching the brick, -1 for missing) and learns, over thousands of
simulated episodes, a strategy that converges to roughly 100% catch rate."

## 2. Core theory

**Reinforcement learning** = an agent takes actions in an environment,
receives rewards, and learns a policy that maximizes cumulative reward,
without being told the "correct" action directly.

- **State (s)**: `(paddle_x, brick_x, brick_y)` — a discretized snapshot
  of the game.
- **Action (a)**: move left / stay / move right.
- **Reward (r)**: +1 catch, -1 miss, -0.01 per step (efficiency nudge).
- **Policy**: implicitly defined by picking the action with the highest
  Q-value in the current state.
- **Episode**: one brick falling from top to bottom.

**Q-learning** is model-free, value-based, and off-policy:
- *Model-free*: never models environment dynamics, only learns from
  experience.
- *Value-based*: learns Q(s,a) = expected future reward of taking action a
  in state s, then acting optimally after. Policy = argmax over actions.
- *Off-policy*: learns the optimal value function even while behaving
  randomly during exploration — this is what makes epsilon-greedy safe.

**Bellman equation (the update rule)**:
```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```
Plain English: nudge the current estimate toward "reward received now, plus
the best value achievable from the next state."

- **alpha (0.1)** = learning rate — how much each experience overwrites
  the old estimate.
- **gamma (0.95)** = discount factor — how much future reward matters vs
  immediate reward. Near 1 = farsighted.
- **epsilon-greedy** = with probability epsilon, take a random action
  instead of the current best guess, to keep exploring. epsilon starts at
  1.0 (fully random) and decays to 0.05 (mostly exploiting, some
  exploration retained).

## 3. Project-specific implementation details

- **State space size**: 10 (paddle_x) x 10 (brick_x) x 10 (brick_y) x 3
  (actions) = 3,000 Q-values, stored as a NumPy array. Small enough for a
  full lookup table — this only works because the state space was
  deliberately discretized to a coarse grid.
- **Files**:
  - `brick_env.py` — pure game logic (state, step, reward), no rendering.
    Mirrors the Gym/Gymnasium `step()` API pattern used in real RL
    libraries.
  - `q_agent.py` — `QLearningAgent` class: action selection, Bellman
    update, epsilon decay, save/load.
  - `train.py` — headless training loop, ~20,000 episodes in under 20
    seconds (no rendering overhead).
  - `play.py` — Pygame visualization, loads the trained Q-table and plays
    greedily (epsilon=0), or `--train` to watch it learn live on-screen.
- **Why separate environment from rendering?** Lets training run fast
  (headless) and reuses identical game logic for the visual demo — no
  duplicated/inconsistent rules between "training mode" and "play mode."
- **Result**: catch rate converges from ~10% (random policy baseline for
  a 10-wide grid) to ~100% after roughly 15,000-20,000 episodes.

## 4. Likely interview questions and how to answer them

**Q: Why Q-learning instead of a neural network / Deep Q-Network (DQN)?**
A: The state space here is tiny and discrete (3,000 states), so a full
lookup table is exact, fast to train, and trivially interpretable — you
can literally print out Q(s,a) for any state. A neural network would be
overkill and slower to converge for a problem this small. DQN earns its
keep when the state space is huge or continuous (e.g. raw pixels), where
a table would be impossibly large.

**Q: How would you scale this to a real Atari game like Breakout?**
A: Real Atari games use raw pixel frames as state, which is a
high-dimensional continuous space — you can't tabulate every possible
pixel configuration. That's exactly what DeepMind's original DQN paper
solved: replace the Q-table with a convolutional neural network that
takes pixels in and outputs Q-values for each action, trained with
experience replay and a target network for stability.

**Q: What's the exploration-exploitation tradeoff and how did you handle it?**
A: If the agent only ever takes the action it currently thinks is best, it
can get stuck on a mediocre strategy and never discover better ones.
Epsilon-greedy balances this: with probability epsilon, take a random
action (explore); otherwise take the best-known action (exploit). Epsilon
starts at 1.0 and decays toward 0.05 over training, so the agent explores
heavily early on and increasingly exploits what it has learned later.

**Q: How do you know it actually learned something, not just memorized?**
A: Evaluated with epsilon=0 (fully greedy, no randomness) on fresh
episodes after training, and measured catch rate — it converges to ~100%,
compared to a random policy's expected ~10% for a 10-wide grid. Since the
state space is small and fully covered during training (every state gets
visited many times across 20k episodes), the Q-table isn't overfitting to
specific runs — it has essentially solved the full state space.

**Q: What are the limitations of this approach?**
A: Tabular Q-learning doesn't scale — the table size grows exponentially
with the number of state variables and their resolution (this is the
"curse of dimensionality"). It also can't generalize: if a state was
never visited during training, its Q-value stays at zero with no
sensible estimate. Function approximation (neural networks, DQN) solves
both problems by learning a continuous function instead of a lookup
table, so it can generalize to unseen states.

**Q: Why the small per-step penalty (-0.01) instead of just +1/-1?**
A: It's a reward-shaping choice: without it, the agent has no signal to
prefer efficient movement over indecisive dithering, since only the final
step matters. A small constant penalty nudges it toward decisive play
without changing which final outcome is optimal.

**Q: What would you add with more time?**
A: A Deep Q-Network to handle continuous/pixel-based state (the natural
next step, matching what real Atari-playing agents use), an actual
brick-breaker/Breakout-style game with a ball and paddle physics instead
of a single falling brick, or comparing against other RL algorithms
(SARSA, policy gradient methods) to reason about the tradeoffs.

## 5. One-line answers to rapid-fire questions

- **On-policy vs off-policy?** Q-learning is off-policy — it learns the
  optimal policy regardless of the exploration policy actually followed.
- **What does the Q-table converge to?** The optimal action-value
  function Q*(s,a) — the true expected return for the best possible play.
- **Why discount future rewards (gamma)?** Encodes that reward now is more
  certain/valuable than reward later, and mathematically guarantees the
  infinite sum of rewards converges.
- **What happens if alpha is too high?** Learning becomes noisy/unstable,
  overwriting good estimates with each new noisy experience.
- **What happens if epsilon decays too fast?** Agent stops exploring
  before it has discovered the optimal policy, and gets stuck exploiting
  a suboptimal one.
