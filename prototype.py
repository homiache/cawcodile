# coding=utf-8

import re
import json
import argparse
import nltk  # nltk is used for count distance between two texts.
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def normalize(text):
    """
    The function removes punctuation from the given string and returns a lowercase string.
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

    # Count of unmatched chars aka "distance" in ML things. :)
    distance = nltk.edit_distance(text1, text2)

    # Get average length of texts
    average_length = (len(text1) + len(text2)) / 2

    # Count percents and return a result
    # TODO Move 0.4 to the config file
    # TODO process ZeroDivisionError
    return distance / average_length < 0.4


def get_intent(text):
    """
    Figure out intentions.
    :param text: string, user input
    :return: string, intent name
    """

    # Check all examples of intents, trying to find a match.
    all_intents = BOT_CONFIG["intents"]
    for intent_name, data in all_intents.items():
        for example in data["examples"]:
            if is_matching(text, example):
                return intent_name

    # Return None if intent is not found
    return None


def get_answer(intent):
    """
    Returns random answer for a given intent
    :param intent: string, intent_name
    :return: string, random answer
    """

    # TODO process invalid value
    responses = BOT_CONFIG["intents"][intent]["responses"]
    return random.choice(responses)


def create_model():
    """
    TODO More description
    TODO Move to a separate script because creation of a model is a rare thing I hope.
    TODO Store already prepared model in a separate file.
    Creates a ML model. The task of the model is to learn how to find intent "y" by input example "x".
    :return: set with sklearn.feature_extraction.text.CountVectorizer \and sklearn.linear_model._logistic.LogisticRegression objects
    """

    # Examples.
    x = []

    # Classes.
    y = []

    # Create two lists of examples and intents.
    for name, data in BOT_CONFIG["intents"].items():
        for example in data['examples']:

            # Get all examples in an one list x.
            x.append(example)

            # Get all intent names as classes in an one list y.
            y.append(name)

    # Creating an object that can transform a set of examples into a set of vectors of numbers
    vectorizer = CountVectorizer()

    # Pass a list of examples so that the vectorizer analyzes them
    vectorizer.fit(x)

    # Transform examples into vectors (sets of numbers)
    x_vectorized = vectorizer.transform(x)

    # Create an object of logistic regression model.
    model = LogisticRegression()

    # Teach the model.
    # After that the model can predict an intent (return predict name) for a given input example.
    model.fit(x_vectorized, y)

    return vectorizer, model


def bot(text):
    """
    The bot actually. Get an input as a text and decides what to do.
    :param text: string
    :return: None
    """

    # Try to find indent
    intent = get_intent(text)

    # If intent is not found try to predict it.
    # Model returns a list of one element with the predicted intent name, so get the first element.
    if not intent:
        intent = model.predict(vectorizer.transform([text]))[0]

    print("(Intent is {})".format(intent))

    # If we have an intent then return a response
    if intent:
        return get_answer(intent)

    # Mock
    # TODO Need to understand if this code will ever be executed. Seems like model never gives up.
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)


def get_args():
    """
    Get values of input args.
    :return: argparse.Namespace, set of named input args.
    """
    parser = argparse.ArgumentParser(description='Bot configuration.')
    parser.add_argument('config', type=str, help='Path to the bot configuration file.')
    return parser.parse_args()


# Just for rough testing
if __name__ == '__main__':

    # Get args
    args = get_args()

    # Load bot config
    # TODO check that file is exist
    # TODO process parsing errors
    # TODO launch documentation
    # TODO Add config structure description
    with open(args.config, "r") as ffile:
        BOT_CONFIG = json.load(ffile)

    # Create the vectorizer and model objects before running the bot
    vectorizer, model = create_model()

    # Unleash the beast!
    while True:
        print(bot(input()))
