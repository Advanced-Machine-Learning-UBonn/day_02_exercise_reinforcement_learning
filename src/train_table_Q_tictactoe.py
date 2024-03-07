"""Train a q-table agent using your TicTacToe board."""

import pickle
from collections import Counter
from typing import Dict, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from gen import Draw, PlayerWins, TicTacToeBoard


# @jax.jit
def create_explore_move(board_array: jnp.ndarray, rng: jnp.ndarray) -> jnp.ndarray:
    """Sample a random move from the set of allowed moves.

    Args:
        board_array (jnp.ndarray): The TicTacToe board as ndarray.
        rng (jax.random.PRNGKey): A random seed.

    Returns:
        jnp.ndarray: The move [x, y] as ndarray.
    """
    noise = jax.random.uniform(rng.squeeze(), (9,), minval=0.0, maxval=1.00)
    allowed = noise - jnp.abs(board_array)
    move = jnp.argmax(allowed)
    move_tuple = jnp.unravel_index(move, (3, 3))
    return jnp.stack(move_tuple)


def process_result(err: Exception) -> Tuple[float, float, str]:
    """Process the game over exception.

    Args:
        err (Exception): An excpetion potentially from four TicTacToeBoard.

    Raises:
        err: Exception from sources other than the board.

    Returns:
        Tuple[float, float, str]: A tuple with the reward for player one,
            the reward for player two, and a string explanation of what happend.
    """
    reward1 = 0.0
    reward2 = 0.0
    event = ""
    if isinstance(err, ValueError):
        incorrect_player = int(str(err)[7])
        if incorrect_player == 1:
            reward1 += -1.0
            reward2 += 0.0
            event = "1 fail"
        else:
            reward1 += 0.0
            reward2 += -1.0
            event = "2 fail"
    elif isinstance(err, PlayerWins):
        winning_player = int(str(err)[7])
        if winning_player == 1:
            reward1 += 1.0
            reward2 += 0.0
            event = "1 won"
        else:
            reward1 += 0.0
            reward2 += 1.0
            event = "2 won"
    elif isinstance(err, Draw):
        reward1 += 0.25
        reward2 += 0.25
        event = "draw"
    else:
        raise err
    return reward1, reward2, event


def board_update(
    board: TicTacToeBoard, move: Tuple[int, int], player: int
) -> Tuple[TicTacToeBoard, Tuple[float, float, Union[None, str]]]:
    """Update the board, handle wins, losses, draws and cheating.

    Args:
        board (TicTacToeBoard): The game board object.
        move (Tuple[int, int]): A move tuple of two integers.
        player (int): The player who made the move, either 1 or 2.

    Returns:
        Tuple[TicTacToeBoard, Tuple[float, float, str]]:
            A tuple of the board, as well as the rewards for
            both player as well as an event description of
            something happend.
            Se the description to None if the board registerd
            the move without an event.
    """
    reward1, reward2, event = 0.0, 0.0, None
    try:
        board.register_move(move, player)
    except Exception as err:
        # set rewards
        reward1, reward2, event = process_result(err)
    return board, (reward1, reward2, event)


if __name__ == "__main__":

    q_table: Dict[str, jnp.ndarray] = {}
    episodes = 10000
    alpha = 0.9  # learning rate
    gamma = 0.9
    epsilon = 0.05
    train_player = 1
    seed = jax.random.key(42)
    outcomes = []
    rewards = []

    bar = tqdm(range(episodes))
    for _ in bar:
        board = TicTacToeBoard()
        state = board.get_board().flatten()
        for i in range(9):
            current_player = i % 2 + 1

            seed = jax.random.split(seed, 1).squeeze()
            if current_player == train_player:
                if str(state) in q_table:
                    if jnp.max(q_table[str(state)] > 0):
                        test_val = jax.random.uniform(seed, (1,))
                        if test_val > epsilon:
                            action_1d = jnp.argmax(q_table[str(state)])
                            action = jnp.stack(jnp.unravel_index(action_1d, (3, 3)))
                        else:
                            action = create_explore_move(state, seed)
                    else:
                        action = create_explore_move(state, seed)
                else:
                    q_table[str(state)] = jnp.zeros(9)
                    action = create_explore_move(state, seed)
                move = action
            else:
                move = create_explore_move(state, seed)

            # register move
            board, (reward1, reward2, event) = board_update(
                board, (int(move[0]), int(move[1])), current_player
            )

            # update table
            new_state = board.get_board().flatten()

            if not str(new_state) in q_table:
                q_table[str(new_state)] = jnp.zeros(9)

            if train_player == 2:
                reward = reward2
            else:
                reward = reward1

            action_pos = jnp.ravel_multi_index((move[0], move[1]), (3, 3))
            update = alpha * (
                reward
                + gamma * jnp.max(q_table[str(new_state)])
                - q_table[str(state)][action_pos]
            )
            q_table[str(state)] = (
                q_table[str(state)] + jax.nn.one_hot(action_pos, 9) * update
            )
            state = new_state

            if event:
                rewards.append(reward)
                outcomes.append(event)
                break

        bar.set_description(
            f"reward: {np.mean(rewards[-1000:]):3.3f}, events: {(Counter(outcomes[-1000:])).most_common()}"
        )

    with open("weights/q_agent.pkl", "wb") as f:
        pickle.dump(q_table, f)
