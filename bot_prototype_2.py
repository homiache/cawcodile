# coding=utf-8

import re
import nltk


class Bot(object):

    """
    Bot as an object.
    """

    __bot_config = None  # JSON data with intents.

    def __init__(self):
        """
        Bot constructor.
        """
        pass

    @staticmethod
    def __normalize(text):
        """
        The function removes punctuation from the given string and returns a lowercase string.
        :param text: string
        :return: string
        """

        # Check input.
        if text.__class__ is not str:
            raise Exception("Input for the method must be a string.")

        # Transform input text to the lowercase.
        text = text.lower()

        # Punctuation to remove.
        # [^ ] matches a single character that is not contained within the brackets.
        # \w means alphanumeric characters plus "_"
        # \s means whitespace characters.
        punctuation = r"[^\w\s]"

        # Remove everything that matches "punctuation to remove" with an empty string "" and return the result.
        return re.sub(punctuation, "", text)

    def __is_matching(self, text1, text2):
        """
        Decide if the texts are similar enough.
        :param text1: string
        :param text2: string
        :return: boolean, Are the texts similar enough?
        """

        # Check inputs.
        if text1.__class__ is not str or text2.__class__ is not str:
            raise Exception("Input for the method must be a string.")

        # Normalize text to avoid false-negative distance.
        text1 = self.__normalize(text1)
        text2 = self.__normalize(text2)

        # Count of unmatched chars aka "distance" in ML things. :)
        distance = nltk.edit_distance(text1, text2)

        # Get average length of texts.

        average_length = (len(text1) + len(text2)) / 2

        # Count percents and return a result.
        # To avoid ZeroDivisionError in case when both texts are empty just return True (empty strings are equal).
        if average_length == 0:
            return True
        else:
            # TODO Move 0.4 to the config file
            return distance / average_length < 0.4

    def __get_intent(self, text):
        """
        Figure out intentions.
        :param text: string, user input
        :return: string, intent name
        """

        # Check input.
        if text.__class__ is not str:
            raise Exception("Input for the method must be a string.")

        # Check that config is present.
        if self.__bot_config is not None:
            # Check all examples of intents, trying to find a match.
            all_intents = self.__bot_config["intents"]
            for intent_name, data in all_intents.items():
                for example in data["examples"]:
                    if self.__is_matching(text, example):
                        return intent_name

        # Return None if intent is not found.
        return None
