# coding=utf-8

import re
import nltk  # nltk is used for count distance between two texts.


def normalize(text):
    """
    Function removes punctuations from a given string and returns the string in lowercase.
    :param text: string
    :return: string
    """

    # Lowercase
    text = text.lower()

    # Punctuation to remove.
    # [^ ] matches a single character that is not contained within the brackets.
    # \w means alphanumeric characters plus "_"
    # \s means whitespace characters.
    punctuation = r"[^\w\s]"

    # Remove everything that matches "punctuation to remove" with an empty string "" and return the result.
    return re.sub(punctuation, "", text)


def is_matching(text1, text2):
    """
    Decide if the texts are similar enough.
    :param text1: string
    :param text2: string
    :return: boolean, Are the texts similar enough?
    """

    # Normalize text to avoid false-negative distance.
    text1 = normalize(text1)
    text2 = normalize(text2)

    # Count of unmatched chars aka distance in ML things. :)
    distance = nltk.edit_distance(text1, text2)

    # Get average length of texts
    average_length = (len(text1) + len(text2)) / 2

    # Count percents and return a result
    # TODO Move 0.4 to the config file
    return distance / average_length < 0.4
