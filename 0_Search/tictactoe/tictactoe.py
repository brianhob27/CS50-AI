"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_num = 0
    o_num = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] == "X":
                x_num += 1
            elif board[i][j] == "O":
                o_num += 1

    if x_num > o_num:
        return "O"
    else:
        return "X"


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    answer = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                ans = (i, j)
                answer.add(ans)

    return answer


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if board[action[0]][action[1]] != EMPTY:
        raise ValueError

    turn = player(board)

    game_board = copy.deepcopy(board)
    game_board[action[0]][action[1]] = turn

    return game_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    win = None

    # Check horizontal
    for i in range(3):
        if board[i][0] != EMPTY:
            if board[i][0] == board[i][1] == board[i][2]:
                win = board[i][0]

    # Check vertical
    if win is None:
        for j in range(3):
            if board[0][j] != EMPTY:
                if board[0][j] == board[1][j] == board[2][j]:
                    win = board[0][j]

    # Check diagonal
    if win is None:
        if board[0][0] != EMPTY:
            if board[0][0] == board[1][1] == board[2][2]:
                win = board[0][0]
    if win is None:
        if board[0][2] != EMPTY:
            if board[0][2] == board[1][1] == board[2][0]:
                win = board[0][2]

    return win


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == "X":
        return 1
    elif winner(board) == "O":
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    ai = player(board)

    answer = ()

    if ai == "X":
        answer = check_max(board)
    elif ai == "O":
        answer = check_min(board)

    return answer[0], answer[1]


def check_min(board):
    min_answer = 2
    min_action = ()

    action_set = actions(board)
    for i in action_set:
        new_board = result(board, i)
        if terminal(new_board):
            answer = utility(new_board)
            if answer < min_answer:
                min_answer = answer
                min_action = i
        else:
            checking = check_max(new_board)
            if checking[2] < min_answer:
                min_answer = checking[2]
                min_action = i
        if min_answer == -1:
            break

    return min_action[0], min_action[1], min_answer


def check_max(board):
    max_answer = -2
    max_action = ()

    action_set = actions(board)
    for i in action_set:
        new_board = result(board, i)
        if terminal(new_board):
            answer = utility(new_board)
            if answer > max_answer:
                max_answer = answer
                max_action = i
        else:
            checking = check_min(new_board)
            if checking[2] > max_answer:
                max_answer = checking[2]
                max_action = i
        if max_answer == 1:
            break

    return max_action[0], max_action[1], max_answer
