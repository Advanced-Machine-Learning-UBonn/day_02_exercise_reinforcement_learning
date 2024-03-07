"""See if our neural agent understand a win condition."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
import pickle
import sys

import numpy as np

sys.path.insert(0, "./src/")

from src.gen import TicTacToeBoard
from src.train_neural_Q_tictactoe import Agent


def test_agent():
    """Test the neural agent.

    Consider the board:

    0|x x  |
    1|  o  |
    2|o    |
      0 1 2

    0 2 is the winning move here for player 1.
    """
    board = TicTacToeBoard()
    move = (0, 0)
    board.register_move(move, 1)
    move = (2, 0)
    board.register_move(move, 2)
    move = (0, 1)
    board.register_move(move, 1)
    move = (1, 1)
    board.register_move(move, 2)

    with open("weights/neural_agent.pkl", "rb") as file:
        model_size, agent_weights = pickle.load(file)

    agent = Agent(model_size)
    probs = agent.apply(agent_weights, board.get_board().flatten(), player=1)
    print(board)
    print(np.array_str(probs, precision=2, suppress_small=True))
    move = np.unravel_index(np.argmax(probs), (3, 3))
    assert move == (0, 2)
