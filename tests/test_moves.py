"""Check the move related functions for the neural agent traning code."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, "./src/")


from src.gen import TicTacToeBoard
from src.train_neural_Q_tictactoe import get_move
from src.train_table_Q_tictactoe import create_explore_move


@pytest.mark.parametrize("move", list((x, y) for x in range(3) for y in range(3)))
def test_get_move(move):
    """Ensure get move returns the maximum."""
    board = TicTacToeBoard()
    board.register_move(move, 1)
    test_move = get_move(board.get_board())
    assert all((m == tm for m, tm in zip(move, test_move)))


@pytest.mark.parametrize("pick", list((x, y) for x in range(3) for y in range(3)))
def test_explore_move(pick):
    """Ensure explore moves are legal."""
    board_array = np.ones((3, 3))
    board_array[pick] = 0
    board_array = jnp.array(board_array)
    explore_move = create_explore_move(board_array.flatten(), jax.random.key(42))
    assert jnp.allclose(jnp.array(pick), explore_move)
