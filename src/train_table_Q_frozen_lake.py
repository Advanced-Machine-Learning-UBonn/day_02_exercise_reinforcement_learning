"""Explore Q-Table optimization using the frozen lake problem.

See also: https://mlabonne.github.io/blog/posts/2022-02-13-Q_learning.html
for a neat blog post explaining this setup.
"""

import time
from collections import Counter

import gymnasium as gym
import numpy as np

# Initialize the non-slippery Frozen Lake environment
environment = gym.make(
    "FrozenLake-v1", is_slippery=False, render_mode="ansi", max_episode_steps=100
)  # ansi
environment.reset()
# environment.render()

state_no = environment.observation_space.n
action_no = environment.action_space.n
q_table = np.zeros((state_no, action_no))

# parameters
episodes = 1000
alpha = 0.9
gamma = 0.9

outcomes = []

for _ in range(episodes):
    state, _ = environment.reset()
    game_over = False
    outcome = "failed"

    while not game_over:

        # TODO: if a q_table entry exists for state and the reward
        #       is not the same for all actions choose the most
        #       likely action.
        #       If no entry exists choose a random action from
        #       'environment.action_space.sample()'.
        action = 0  # TODO: remove me.

        # ask for environment feedback.
        new_state, reward, game_over, info, _ = environment.step(action)

        # update the Q-Table
        # TODO: compute the Q-Table update.

        state = new_state

        if reward:
            outcome = "win"
        if environment._elapsed_steps > 1000:
            game_over = True

    outcomes.append(outcome)

    print(outcomes)
    print(Counter(outcomes))

test_env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
state, _ = test_env.reset()
game_over = False
while not game_over:
    action = np.argmax(q_table[state])
    state, _, game_over, _, _ = test_env.step(action)
    time.sleep(1)
