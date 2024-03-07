"""This module ships a function."""

import os
from typing import Any, Callable, Dict, Tuple, Union

os.environ["JAX_PLATFORMS"] = "cpu"

import pickle

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen import Module

from gen import TicTacToeBoard
from train_neural_Q_tictactoe import Agent, get_move


def neural_agent_forward(
    weights: FrozenDict, board_array: jnp.ndarray, player: int, agent: Module
) -> Tuple[jnp.ndarray, ...]:
    """Get a move from a neural agent.

    Args:
        weights (FrozenDict): The agent's weights.
        board_array (jnp.ndarray): The game's board as ndarray.
        player (int): The player we are playing with.
        agent (Module): The flax Module for the agent.

    Returns:
        [jnp.ndarray, jnp.ndarray]: Tuple with the agents next move coordinates,
            i.e. [1 1].
    """
    move_probs = agent.apply(weights, board_array.flatten(), player=player)
    move = get_move(move_probs)
    return move


def q_table_agent_forward(
    q_table: Dict[str, jnp.ndarray], board_array: jnp.ndarray, player: int, agent: None
) -> Tuple[jnp.ndarray, ...]:
    """Get a move from a q-table.

    Args:
        q_table (Dict[str, jnp.ndarray]): A dictionary with the next move probability
            distributions.
        board_array (jnp.ndarray): The board as ndarray.
        player (int): Not used, here for compatability with neural agent.
        agent (None): Not used, here for compatability with neural agent.

    Returns:
        [jnp.ndarray, jnp.ndarray]: Tuple with the agents next move coordinates,
            i.e. [1 1].
    """
    state = str(board_array.flatten())
    action = jnp.argmax(q_table[str(state)])
    move = jnp.unravel_index(action, (3, 3))
    return move


if __name__ == "__main__":
    load_neural_agent = False

    if load_neural_agent is True:
        print("loading neural agent")
        with open("weights/neural_agent.pkl", "rb") as file:
            model_size, agent_weights = pickle.load(file)

        agent = Agent(model_size=model_size)

        print("db0 norm", jnp.linalg.norm(agent_weights["params"]["Dense_0"]["bias"]))
        print("dk0 norm", jnp.linalg.norm(agent_weights["params"]["Dense_0"]["kernel"]))
        print("db1 norm", jnp.linalg.norm(agent_weights["params"]["Dense_1"]["bias"]))
        print("dk1 norm", jnp.linalg.norm(agent_weights["params"]["Dense_1"]["kernel"]))
        print("db2 norm", jnp.linalg.norm(agent_weights["params"]["Dense_2"]["bias"]))
        print("dk2 norm", jnp.linalg.norm(agent_weights["params"]["Dense_2"]["kernel"]))
        print("db2 raw", agent_weights["params"]["Dense_2"]["bias"].reshape((3, 3)))
        forward: Callable[[Any, Any, int, Any], Tuple[jnp.ndarray, ...]] = (
            neural_agent_forward
        )
    else:
        print("loading q_table")
        with open("weights/q_agent.pkl", "rb") as file:
            agent_weights = pickle.load(file)
        agent = None
        forward = q_table_agent_forward

    move: Union[Tuple[jnp.ndarray, ...], Tuple[int, ...]]

    board = TicTacToeBoard()
    print("Choose the game mode:")
    print("1 - AI versus AI")
    print("2 - Human versus AI")
    choice = int(input("Choose 1 or 2."))
    if choice == 1:
        try:
            for i in range(9):
                player = i % 2 + 1
                current_board_array = board.get_board()
                move = forward(agent_weights, board.get_board(), player, agent)
                print(f"round: {i}, player {player}: {int(move[0]), int(move[1])}")
                board.register_move(move, player)
                print(board)
        except Exception as err:
            print(err)
    elif choice == 2:
        print("Player 1 or 2?")
        human_player = int(input("Choose 1 or 2."))
        print('Enter the array-indices of your move i.e. " 1 1 ".')
        try:
            for i in range(9):
                player = i % 2 + 1
                if player == human_player:
                    print(board)
                    moved = False
                    while not moved:
                        try:
                            user_move_str = input("Make your move:")
                            move = tuple(int(m) for m in user_move_str.split())
                            moved = True
                        except Exception as err:
                            print(err)
                            moved = False
                else:
                    move = forward(agent_weights, board.get_board(), player, agent)

                print(f"round: {i}, player {player}: {int(move[0]), int(move[1])}")
                board.register_move(move, player)
        except Exception as err:
            print(type(err), err)

    else:
        print("Game mode unkown.")

    print(board)
