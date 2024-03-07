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
        if player not in (1, 2):
            raise ValueError("Players should be 1 or 2.")

        if self.board[pos] == 0:
            self.board[pos] = 1 if player == 1 else -1
        else:
            raise ValueError(f"Player {player} chose an occupied board position.")

        self._check_win()

    def _check_win(self):
        """Evaluate the win and draw conditions.

        Raises:
            PlayerWins: If a player won.
            Draw: If the board is full.
        """
        for player in (1, 2):
            check = 1 if player == 1 else -1
            col_sum = np.sum(self.board == check, axis=0)
            row_sum = np.sum(self.board == check, axis=1)
            diag_sum = np.sum(np.diag(self.board) == check)
            ud_diag_sum = np.sum(np.diag(np.fliplr(self.board)) == check)

            if (
                any(col_sum == 3)
                or any(row_sum == 3)
                or diag_sum == 3
                or ud_diag_sum == 3
            ):
                raise PlayerWins(f"Player {player} wins.")

            if all((self.board != 0).flatten()):
                raise Draw("It's a draw.")

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
