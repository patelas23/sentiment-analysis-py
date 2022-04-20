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

import sys
import re
import itertools

from collections import defaultdict


def main():

    training_file_name = sys.argv[1]
    # test_file_name = sys.argv[2]
    # training_model_file_name = sys.argv[3]

    # Input buffers
    training_text = ""
    test_text = ""
    model_text = ""
    with open(training_file_name) as f:
        training_text = f.read()

    print(extract_context(training_text))


# Function which counts each occurrence of a word
#     with a given sentiment
def train_model(corpus_text):
    model_dict = defaultdict()
    context_lines = extract_context(corpus_text)
    sentiment_lines = extract_sentiment(corpus_text)

    positive_dict = defaultdict()
    negative_dict = defaultdict()

    positive_count = 0
    negative_count = 0

    # Iterate over both lists in parallel
    for context, sentiment in zip(context_lines, sentiment_lines):
        current_line = context.split()
        if(sentiment == "positive"):
            count_features(current_line, positive_dict)
            positive_count += 1
        else:
            count_features(current_line, negative_dict)
            negative_count += 1

    # Sort each sentiment dictionary
    #
    # Calculate log-likelihood of each word for each sentiment


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


# Function for determining the log-likelihood in the supplied dictionary
def calculate_likelihood():
    pass


# Function to combine sentiment dictionaries, removing overlaps
#     where the likelihood of one is higher than the other.
def remove_overlaps(negative_bag, positive_bag):
    pass


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
            tup_sentiment = model_tup[0]
            tup_likelihood = model_tup[1]

            f.write("Word: " + word + "; Sentiment: " + tup_sentiment + 
                             "; Likelihood: " + tup_likelihood)


if __name__ == "__main__":
    main()
