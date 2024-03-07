# Exercise Reinforcement Learning
This exercise explores Q-learning based on neural and table-based approaches. Use Vscodes preview for 
a proper rendering of the math.

### Q-Table based RL:
#### Frozen-Lake:

1.1 Take a look at the [frozen-lake-documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/), and familiarize yourself with the setup.
- Recall the Q-Table update Rule we saw during the lecture or take a look at [1, page 153]:


$$
    \begin{align}
    Q(s_t, a_t)_{\text{update}} = Q(s_t, a_t) + \alpha (r_t  + \gamma \max_a (Q(s_{t + 1}, a)) - Q(s_t, a_t) ) 
    \end{align}
$$

Here $Q \in \mathbb{R}^{N_s, N_a}$ denotes the Q-Table with $N_s$ the number of states and $N_a$ the number of possible actions. Reward at time t $r_t$ learning rate $\alpha$ as well as the discount factor $\gamma$.

1.2 Navigate to `src/train_table_Q_frozen_lake.py`, resolve the `TODO`s by choosing actions from a Q-Table and implementing the Q-Table update rule.

If you see your figure reach the goal in the short movie clip you are done.

#### Tic-Tac-Toe
2.1 Let's consider a more challenging example next. Navigate to `src/train_table_Q_tictactoe.py` and finish the `TicTacToeBoard` class in `gen.py`. Use `nox -s test` to check your
progress. When `tests/test_board.py` checks out without an error, this task is done.

2.2 Set up a Q-Table-based agent and train it. To do so open `src/train_table_Q_tictactoe.py` and use a Python dictionary as your data structure for Q-Table. The dictionary will allow you to encode the board states as strings.

2.3 Implement the `create_explore_move` and `process_result` functions. Use `tests/test_explore_move.py` to test your code. 

2.4 Recycle your code from the frozen lake example to implement Q-Table (or dict) learning. Implement learning by playing games against a random opponent. The opponent performs random moves all the time. Use your `create explore move` function for the opponent. 

2.5 Stop sampling from the Q-table every time for your agent. Sample the agent with probability $\epsilon$ and
perform an exploratory move with probability $1 - \epsilon$. Rember to use `jax.random.split` to generate new seeds for `jax.random.uniform`. 

### Q-Neural RL:
3.1 Let's replace the Table with a neural network [2, Algorithm 1]. For simplicity, we do not implement batching, this works here, but won't scale to harder problems. Re-use as much code from task two as possible. Open  `src/train_neural_Q_tictactoe.py`. Your `board_update`, `create_explore_move` functions are already imported.
Recall the cost function for neural Q-Learning:

$$
    \begin{align}
        L(\theta) = \frac{1}{N_a \sum_{i=1}^{N_a} (y_i - Q_n(s,a; \theta)_i)^2
    \end{align}
$$

with $Q_n(s,a; \theta) \approx Q(s,a)$ the neural Q-Table approximator. And $\mathbf{y}$ the desired output 
at the current optimization step. Construct $\mathbf{y} \in \mathbb{R}^{3 \cdot 3}$ by inserting 

$$
\begin{align}
\mathcal{y} =
\begin{cases}
    r,  & \text{  if the game ended} \\
    r + \gamma \max_a Q(s_{t+1, a; \theta}) & \text{ else}
\end{cases}
\end{align}
$$

into $\mathbf{y}$ the the position of the move taken. Compute gradients using `jax.grad` and update your weights using `jax.tree_map(lambda w, g: w - alpha*g, weights, grads)`.

#### Play against your agent
`play.py` allows playing against the trained TicTacToe agents.

### Optional further reading:
- [1] Sutton, Reinforcement Learning, https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
- [2] Playing Atari with Deep Reinforcement Learning, https://arxiv.org/pdf/1312.5602.pdf
- [3] Pong from Pixels, https://karpathy.github.io/2016/05/31/rl/
