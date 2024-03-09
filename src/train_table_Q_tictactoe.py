"""Train a q-table agent using a TicTacToe board."""

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
    # TODO: find an explorative move from the set of allowed moves given the
    # state of the game in the board_array.
    move_tuple = (0, 0)
    return jnp.stack(move_tuple)


def process_result(err: Exception) -> Tuple[float, float, str]:
    """Process the game over exception.

    Args:
        err (Exception): An excpetion potentially from your TicTacToeBoard.

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
        # TODO: change the following line according to your
        # error message to extract the incorrect player.
        incorrect_player = int(str(err)[7])
        # TODO: set reward1, reward2 and event
        # for the cheating case.

    elif isinstance(err, PlayerWins):
        # TODO: change the following line according to your
        # error message to extract the winning player.
        winning_player = int(str(err)[7])
        if winning_player == 1:
            # TODO: set reward1, reward2 and event
            # for the player1 wins case.
            pass
        else:
            # TODO: set reward1, reward2 and event
            # for the player2 wins case.
            pass
    elif isinstance(err, Draw):
        # TODO: set reward1, reward2 and event
        # for the draw case.
        pass
    else:
        # A different expection appeared, pass it along.
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
            Set the description to None if the board registerd
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
    outcomes = []  # append event information strings here.
    rewards = []  # record reward values here

    bar = tqdm(range(episodes))
    for _ in bar:
        board = TicTacToeBoard()
        state = board.get_board().flatten()
        for i in range(9):
            current_player = i % 2 + 1
            seed = jax.random.split(seed, 1).squeeze()  # Create a new random seed.

            # TODO: Implement Q-learning for the TicTacToe game:

            # TODO: Set the next move either using the best existing action
            # for the current state or randomly using create_explore_move

            # TODO: Register a new move using board_update().
            event = ""
            reward = 0

            # TODO: Get the new state and save it to the q-table.

            # TODO: Update the table using the q-table-update rule.

            # TODO: Set the new state for the next iteration.

            if event:
                rewards.append(reward)
                outcomes.append(event)
                break

        bar.set_description(
            f"reward: {np.mean(rewards[-1000:]):3.3f}, events: {(Counter(outcomes[-1000:])).most_common()}"
        )

    with open("weights/q_agent.pkl", "wb") as f:
        pickle.dump(q_table, f)
