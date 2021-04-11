"""
Different utility functions
"""

from utils.utils_data import action_json

COLOURS = ['red', 'green', 'blue', 'yellow']
VALUES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Skip', 'Reverse',
          'Draw Two', 'Wild', 'Wild Draw Four']


def get_card_names(hand_cards):
    """
    Return verbosed hand cards. Example input (from trajectory observation):
    [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    :param hand_cards:
    :return:
    """

    hand = ""
    for idx, row in enumerate(hand_cards):
        row_cards = [i for i, e in enumerate(row) if e == 1]
        if len(row_cards) > 0:
            for card in row_cards:
                hand += COLOURS[idx] + "-" + VALUES[card] + " | "

    return hand


def get_action_names(action_number):
    """
    Function to return the name of an action given the
    :param action_number:
    :return
    """
    return list(action_json.keys())[list(action_json.values()).index(action_number)]


get_action_names(60)
