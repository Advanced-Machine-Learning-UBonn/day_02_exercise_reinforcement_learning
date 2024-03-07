"""Implement a TicTacToe Board."""

from typing import Tuple

import jax.numpy as jnp
import numpy as np


class PlayerWins(Exception):  # noqa N818
    """Thrown is a player wins."""

    pass


class Draw(Exception):  # noqa N818
    """Thrown is the game is a draw."""

    pass


class TicTacToeBoard(object):
    """Implement TicTacToe board and win conditions."""

    def __init__(self):
        """Set up the board."""
        self.board = np.zeros([3, 3])

    def register_move(self, pos: Tuple[int, int], player: int):
        """Register a player's move.

        Args:
            pos (Tuple[int, int]): A (row, column) coordinate tuple.
            player (int): An integer designating who is making the move.

        Raises:
            ValueError: If the player names are not 1 or 2,
                or if a position on he board is already occupied.
        """
        # TODO: implement me.

        self._check_win()

    def _check_win(self):
        """Evaluate the win and draw conditions.

        Players with 3 board positions in a row win.
        Check the row, column and diagonal win conditions.

        If no player won and the board is full its a draw.

        Raises:
            PlayerWins: If a player won.
            Draw: If the board is full.
        """
        # TODO: implement me.

    def __repr__(self):
        """Implement how this object behaves within print statements."""
        out_array = np.array(self.board).astype(str)
        out_array[self.board == 1] = "x"
        out_array[self.board == -1] = "o"
        out_array[self.board == 0] = ""
        return str(out_array)  # .replace('\'', '  ')

    def get_board(self) -> jnp.ndarray:
        """Return the current state of the board as array."""
        return jnp.array(self.board)
