"""Train a neural Q agent."""

import pickle
from collections import Counter
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from tqdm import tqdm

from gen import TicTacToeBoard
from train_table_Q_tictactoe import board_update, create_explore_move


def flip(x: int) -> int:
    """Flip the board for player 2."""
    return 3 - 2 * x


class Agent(nn.Module):
    """Define a neural q agent Module."""

    model_size: int

    @nn.compact
    def __call__(self, board: jnp.ndarray, player: int) -> jnp.ndarray:
        """Compute the neural agents forward pass.

        Args:
            board (jnp.ndarray): The boards current state as ndarray.
            player (int): The player designation. Either 1 or 2.

        Returns:
            jnp.array: The reward prediction for each allowed move.
        """
        fboard = board * flip(player)
        hidden = nn.relu(nn.Dense(self.model_size, use_bias=True)(fboard))
        hidden = nn.relu(nn.Dense(self.model_size, use_bias=True)(hidden))
        out = nn.Dense(9, use_bias=True)(hidden)
        out = nn.sigmoid(out)
        out = out - jnp.abs(board)
        return jnp.reshape(out, (3, 3))  # the probabilities for each move.


@jax.jit
def get_move(move_probs: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    """Get the most promising move from a reward prediction array.

    Args:
        move_probs (jnp.ndarray): The reward predictions for each possible move.

    Returns:
        Tuple[jnp.ndarray, ...]: Move coordinates in 2D. I.e. [0 1].
    """
    move = jnp.argmax(move_probs)
    move_2d = jnp.unravel_index(move, (3, 3))
    return move_2d


@jax.jit
def set_up_gt(move: Tuple[jnp.ndarray, jnp.ndarray], desired: float) -> jnp.ndarray:
    """Set up the ground truth for backpropagation.

    Args:
        move (Tuple[jnp.ndarray, jnp.ndarray]): The potentially winning move.
        desired (float): The desired output y.

    Returns:
        jnp.ndarray: A ground truth array of shape (3, 3).
    """
    gt = jnp.zeros((3, 3)).at[move[0], move[1]].set(desired)
    return gt


@jax.jit
def cost(
    weights: FrozenDict, y: jnp.ndarray, move: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """Compute the cost to enable backprop into the neual Q agent.

    Args:
        weights (FrozenDict): The agents network weights.
        y (jnp.ndarray): The desired output y.
        move (Tuple[jnp.ndarray, jnp.ndarray]): The move we made.

    Returns:
        jnp.ndarray: The agent's reward prediction error.
    """
    y = set_up_gt(move, y)
    return jnp.mean((y - agent.apply(weights, state, player=current_player)) ** 2)


if __name__ == "__main__":

    q_table = {}
    episodes = 5000
    alpha = 0.9  # learning rate
    gamma = 0.9
    epsilon = 0.05
    train_player = 1
    seed = jax.random.key(42)
    outcomes = []
    rewards = []

    model_size = 36
    load = False

    agent = Agent(model_size=model_size)

    rng = jax.random.key(42)
    if not load:
        agent_weights = agent.init(
            rng, TicTacToeBoard().get_board().flatten(), player=1
        )
    else:
        with open("weights/neural_agent.pkl", "rb") as file:
            agent_weights, _ = pickle.load(file)

    bar = tqdm(range(episodes))
    for _ in bar:
        board = TicTacToeBoard()
        state = board.get_board().flatten()
        for i in range(9):
            current_player = i % 2 + 1

            seed = jax.random.split(seed, 1).squeeze()
            if current_player == train_player:
                test_val = jax.random.uniform(seed, (1,))
                if test_val > epsilon:
                    probs = agent.apply(agent_weights, state, player=current_player)
                    action_1d = jnp.argmax(probs)  # type: ignore
                    action = jnp.unravel_index(action_1d, (3, 3))
                else:
                    action = create_explore_move(state, seed)
                move = action
            else:
                move = create_explore_move(state, seed)

            # register move
            board, (reward1, reward2, event) = board_update(
                board, (move[0], move[1]), current_player
            )

            # update table
            new_state = board.get_board().flatten()

            if not str(new_state) in q_table:
                q_table[str(new_state)] = jnp.zeros(9)

            if train_player == 2:
                reward = reward2
            else:
                reward = reward1

            if event:
                y = reward
            else:
                probs = agent.apply(agent_weights, new_state, player=current_player)
                y = reward + gamma * jnp.max(probs)  # type: ignore

            grad_cost = jax.grad(cost)
            grads = grad_cost(agent_weights, y, move)

            # update =  alpha * (
            #     reward + gamma * jnp.max(q_table[str(new_state)]) - q_table[str(state)][action]
            # )
            # q_table[str(state)] = q_table[str(state)] + jax.nn.one_hot(action, 9)*update
            agent_weights = jax.tree_map(
                lambda w, g: w - alpha * g, agent_weights, grads
            )

            state = new_state

            if event:
                rewards.append(reward)
                outcomes.append(event)
                break

        bar.set_description(
            f"reward: {np.mean(rewards[-1000:]):3.3f}, events: {(Counter(outcomes[-1000:])).most_common()}"
        )

    with open("weights/neural_agent.pkl", "wb") as f:
        pickle.dump([model_size, agent_weights], f)
