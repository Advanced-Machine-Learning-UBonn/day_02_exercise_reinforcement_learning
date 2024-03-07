# Exercise Reinforcement Learning
This exercise explores neural and table-based Reinforcement Learning.

#### Q-Table based RL:
##### Frozen-Lake:

1.1 Take a look at the [frozen-lake-documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/), familiarize yourself with the setup.
    - Recall the Q-Table update Rule we saw during the lecture or take a look at [1, page 153]:

$$  Q(s_t, a_t)_{\text{update}} = Q(s_t, a_t) + \alpha (r_t  + \gamma \max_a(Q(s_{t+1}, a)) - Q(s_t, a_t) ) $$

Here $Q \in \mathbb{R}^{N_s, N_a}$ denotes the Q-Table with $N_s$ the number of states and $N_a$ the number of possible actions. Reward at time t $r_t$ learning rate $\alpha$ as well as the 

1.2 Navigate to `src/train_table_Q_frozen_lake.py`, resolve the `TODO`s by choosing actions from a Q-Table and implementing the Q-Table update rule.

If you see your figure reach the goal in the short movie clip you are done.

#### Tic-Tac-Toe
2.1 Let's consider a more challenging example next. Navigate to `src/train_table_Q_tictactoe.py` and finish the `TicTacToeBoard` class in `gen.py`. Use `nox -s test` to check your
progress. When `tests/test_board.py` checks out without an error, this task is done.

2.2 

#### Q-Neural RL:
    - `src/train_neural_Q_tictactoe.py`

`play.py` allows playng against the trained TicTacToe agents.

### Further Reading:
- [1] Sutton, Reinforcement Learning, https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
- [2] Playing Atari with Deep Reinforcement Learning, https://arxiv.org/pdf/1312.5602.pdf
- [3] Pong from Pixels, https://karpathy.github.io/2016/05/31/rl/
