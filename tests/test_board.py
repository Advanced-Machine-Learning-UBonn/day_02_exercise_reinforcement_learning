"""Test the python function from src."""

import sys

import pytest

sys.path.insert(0, "./src/")

from src.gen import Draw, PlayerWins, TicTacToeBoard


@pytest.mark.parametrize("player", (1, 2))
@pytest.mark.parametrize("pos", (0, 1, 2))
def test_wins(player: int, pos: int) -> None:
    """Test row and column wins."""
    board = TicTacToeBoard()
    with pytest.raises(PlayerWins, match=f"Player {player} wins."):
        board.register_move((pos, 0), player)
        board.register_move((pos, 1), player)
        board.register_move((pos, 2), player)

    board = TicTacToeBoard()
    with pytest.raises(PlayerWins, match=f"Player {player} wins."):
        board.register_move((0, pos), player)
        board.register_move((1, pos), player)
        board.register_move((2, pos), player)


@pytest.mark.parametrize("player", (1, 2))
def test_wins_diag(player: int) -> None:
    """Test diagnoal wins."""
    board = TicTacToeBoard()
    with pytest.raises(PlayerWins, match=f"Player {player} wins."):
        board.register_move((0, 0), player)
        board.register_move((1, 1), player)
        board.register_move((2, 2), player)

    board = TicTacToeBoard()
    with pytest.raises(PlayerWins, match=f"Player {player} wins."):
        board.register_move((0, 2), player)
        board.register_move((1, 1), player)
        board.register_move((2, 0), player)


def test_draw() -> None:
    """Test a draw game."""
    board = TicTacToeBoard()
    with pytest.raises(Draw):
        board.register_move((1, 1), 1)
        board.register_move((2, 2), 2)

        board.register_move((0, 1), 1)
        board.register_move((2, 1), 2)

        board.register_move((0, 0), 1)
        board.register_move((0, 2), 2)

        board.register_move((2, 0), 1)
        board.register_move((1, 0), 2)

        board.register_move((1, 2), 1)


@pytest.mark.parametrize("player", (0, 3, 4))
def test_player_not_exist(player: int) -> None:
    """Test incorrect player designations."""
    board = TicTacToeBoard()
    with pytest.raises(ValueError):
        board.register_move((1, 1), player)


def test_position_occupied() -> None:
    """Test position overlaod."""
    board = TicTacToeBoard()
    with pytest.raises(ValueError):
        board.register_move((1, 1), 0)
        board.register_move((1, 1), 0)
