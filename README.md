# Exercise Reinforcement Learning
This exercise explores Q-learning based on neural and table-based approaches. 
## Task 1: Q-Table based RL - Frozen-Lake

1. Take a look at the [frozen-lake-documentation](https://gymnasium.farama.org/environments/toy_text/frozen_lake/), and familiarize yourself with the setup. Recall the Q-Table update Rule we saw during the lecture or take a look at [1, page 153]:

```math
    \begin{align}
        Q(s_t, a_t)_{\text{update}} = Q(s_t, a_t) + \alpha (r_t  + \gamma \max_a (Q(s_{t+1}, a)) - Q(s_t, a_t) ) 
    \end{align}
```
Here $Q \in \mathbb{R}^{N_s, N_a}$ denotes the Q-Table with $N_s$ the number of states and $N_a$ the number of possible actions, reward $r_t$ at time t, learning rate $\alpha$ as well as the discount factor $\gamma$.

2. Navigate to `src/train_table_Q_frozen_lake.py`, resolve the `TODO`s by choosing actions from a Q-Table and implementing the Q-Table update rule.

3. Execute the script with `python src/train_table_Q_frozen_lake.py`. If you see your figure reach the goal in the short movie clip you are done.

Note: If you run into a `libGL` error, this is usually caused by the conda installation. To fix this, execute the following script in your terminal:
```bash
cd /home/$USER/miniconda3/lib
mkdir backup  # Create a new folder to keep the original libstdc++
mv libstd* backup  # Put all libstdc++ files into the folder, including soft links
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6  ./ # Copy the c++ dynamic link library of the system here
ln -s libstdc++.so.6 libstdc++.so
ln -s libstdc++.so.6 libstdc++.so.6.0.19
```
You might have to adjust the miniconda lib path to your current environment, e.g. if you are using a custom environment named `(aml)` adjust the first command to
```bash
cd /home/$USER/miniconda3/envs/aml/lib
```
Source: [Stackoverflow](https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris)

## Task 2: Q-Table based RL - Tic-Tac-Toe
1. Let's consider a more challenging example next. Navigate to `src/train_table_Q_tictactoe.py` and finish the `TicTacToeBoard` class in `gen.py`. Use `nox -s test` to check your
progress. When `tests/test_board.py` checks out without an error, this task is done.

2. Set up a Q-Table-based agent and train it. To do so open `src/train_table_Q_tictactoe.py` and start with the main routine. Use a Python dictionary as your data structure for the Q-Table. The dictionary will allow you to encode the board states as strings and access them eaisly with `q_table[str(state)]`. Use `create_explore_move` when sampling random moves from the set of allowed moves.

3. Implement the `create_explore_move` and `process_result` functions. Use `tests/test_explore_move.py` to test your code. 

4. Recycle your code from the frozen lake example to implement Q-Table (or dict) learning. Implement learning by playing games against a random opponent. The opponent performs random moves all the time. Use your `create_explore_move` function for the opponent. 

5. Stop sampling from the Q-table every time for your agent. Sample the agent with probability $\epsilon$ and
perform an exploratory move with probability $1 - \epsilon$. Remember to use `jax.random.split` to generate new seeds for `jax.random.uniform`. 

## Task 3: Q-Neural RL - Tic-Tac-Toe
1. Let's replace the Q-table with a neural network [2, Algorithm 1]. For simplicity, we do not implement batching, this works here, but won't scale to harder problems. Re-use as much code from task two as possible. Open  `src/train_neural_Q_tictactoe.py`. Your `board_update`, `create_explore_move` functions are already imported.
Recall the cost function for neural Q-Learning:

$$
    \begin{align}
        L(\theta) = \frac{1}{N_a} \sum_{i=1}^{N_a} (y_i - Q_n(s,a; \theta)_i)^2
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

into $\mathbf{y}$ the position of the move taken. Compute gradients using `jax.grad` and update your weights using `jax.tree_map(lambda w, g: w - alpha*g, weights, grads)`.

### Play against your agent
`play.py` allows playing against the trained TicTacToe agents.

## Optional further reading:
- [1] Sutton, Reinforcement Learning, https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf
- [2] Playing Atari with Deep Reinforcement Learning, https://arxiv.org/pdf/1312.5602.pdf
- [3] Pong from Pixels, https://karpathy.github.io/2016/05/31/rl/
