# sentiment.py
# Author: Ankur Patel
# Course: Natural Language Processing
# CourseID: CMSC416-SP2022
# Instructor: Caroline Budwell
#
# A decision-list based sentiment classifier for
# identifying sentiment from a given sample of tweets
#
# Usage: python3 sentiment.py training_file.txt test_file.txt model_file.txt
#         > output_answers.txt
#
# IN: training_file: set of tweets annotated with sentiment classification
#       test_file: set of tweets to be analyzed for sentiment
#     model_file: name of file to contain sentiment statistics
# OUT: Trained sentiment classifier

import math
import sys
import re
import itertools

from collections import defaultdict

from sklearn.covariance import log_likelihood


def main():

    training_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    training_model_file_name = sys.argv[3]

    # Input buffers
    training_text = ""
    test_text = ""
    model_text = ""
    with open(training_file_name) as f:
        training_text = f.read()

    model_dict = train_model(training_text)
    write_model_to_file(model_dict, training_model_file_name)


def train_model(corpus_text):
    vocabulary = defaultdict(int)
    sentiment_dict = defaultdict(int)

    context_lines = extract_context(corpus_text)
    sentiment_values = extract_sentiment(corpus_text)

    for context, sentiment in zip(context_lines, sentiment_values):
        current_line = context.split()
        for word in current_line:
            # Raw word count
            if word in vocabulary:
                vocabulary[word] += 1
            else:
                vocabulary[word] = 1
            # Count of word given sentiment
            feature_tup = tuple(word, sentiment)
            if feature_tup in sentiment_dict:
                sentiment_dict[feature_tup] += 1
            else:
                sentiment_dict[feature_tup] = 1

    model_dict = calculate_discrimination(vocabulary, sentiment_dict)
    return model_dict



def apply_model(test_corpus):
    test_lines = extract_context(test_corpus)


# Function which populates supplied dictionary with the count of each feature
def count_features(context_line, feature_dict):
    current_line = context_line.split()
    for word in current_line:
        if word in feature_dict:
            feature_dict[word] += 1.0
        else:
            feature_dict[word] = 1.0


# Calculates the ratio of positive to negative sentiment for each word,
# IN:
#    vocab: dictionary containing all parsed words and their associated counts
#    features: dictionary containing count of each instance of a word with a
#        certain sentiment
def calculate_discrimination(vocab, features):
    likelihood_dict = defaultdict()
    for word in vocab:
        negative_word = (word, "negative")
        positive_word = (word, "positive")

        if positive_word in features:
            positive_count = float(features[positive_word])
        else:
            positive_count = 1.0

        if negative_word in features:
            negative_count = float(features[negative_word])
        else:
            negative_count = 1.0

        positive_prob = float(positive_count/vocab[word])
        negative_prob = float(negative_count/vocab[word])

        discriminator = math.log(positive_prob/negative_prob)

        if (discriminator < 0):
            likelihood_dict[word] = (math.fabs(discriminator), "negative")
        else:
            likelihood_dict[word] = (math.fabs(discriminator), "positive")

        # likelihood_dict[word] = (discriminator)

    # Sort dictionary by total level of discrimination
    sorted_likelihood = dict(sorted(likelihood_dict.items(),
                                    key=lambda x: x[1][0]))

    # return likelihood_dict
    return sorted_likelihood


# Helper function returns a list of lines from corpus
#     demarcated by:
#         <context> This is a tweet. </context>
def extract_context(corpus_text):
    return re.findall(r'<context>\s+(.*)\s+</context>', corpus_text)


# Helper function returns a list of sentiment tags
#     corresponding to lines of context
def extract_sentiment(corpus_text):
    return re.findall(r'sentiment="(\S+)"/>', corpus_text)


def clean_text(corpus_text):
    return re.sub(r'', "", corpus_text)


# Method which writes calculated model to file
def write_model_to_file(model, file_path):
    with open(file_path) as f:
        for word in model:
            model_tup = model[word]
            tup_likelihood = model_tup[0]
            tup_sentiment = model_tup[1]

            f.write("Word: " + word + "; Sentiment: " + tup_sentiment +
                    "; Likelihood: " + tup_likelihood)


if __name__ == "__main__":
    main()
