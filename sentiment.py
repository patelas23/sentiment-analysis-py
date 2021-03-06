# sentiment.py
# Author: Ankur Patel
# Course: Natural Language Processing
# CourseID: CMSC416-SP2022
# Instructor: Bridgette Mckinnes
#
# A decision-list based sentiment classifier for
# identifying sentiment from a given sample of tweets
#
# USAGE: python3 sentiment.py training_file.txt test_file.txt model_file.txt
#         > output_answers.txt
#
# IN: training_file: set of tweets annotated with sentiment classification
#       test_file: set of tweets to be analyzed for sentiment
#     model_file: name of file to contain sentiment statistics
# OUT: Trained sentiment classifier
# 
# ALGORITHM: 
#     1. Parse input text into list of lines and their sentiment
#     2. Count each instance of a word with its lines sentiment
#     3. Calculate log-likelihood of each word:sentiment relation
#     4. Parse test file
#     5. Test input against three given base cases
#     6. Test input against each sentiment in model
#     7. Log trained model and answers to their respective files
#
# ACCURACY:
#     most frequent sense: 0.6866
#     sentiment.py: 0.6853

import math
import sys
import re

from collections import defaultdict


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

    with open(test_file_name) as f:
        test_text = f.read()

    apply_model(test_text, model_dict)


# Counts each occurence of a word, and its word-sentiment pair from the given
#    text, and returns a trained classifier 
# IN: 
#    corpus_text - string containing sentiment-annotated tweets
# OUT:
#    model_dict - dictionary of the form {word: (discrimination, sentiment)}
def train_model(corpus_text):
    vocabulary = defaultdict(int)
    sentiment_dict = defaultdict(int)

    context_lines = extract_context(corpus_text)
    sentiment_values = extract_sentiment(corpus_text)
    
    corpus_count = len(context_lines)
    
    negative_count = 0.0
    positive_count = 0.0
    mfs_count = 0.0

    for context, sentiment in zip(context_lines, sentiment_values):
        current_line = context.split()
        # Count each sense for mfs calculation
        if sentiment == "positive":
            positive_count += 1.0
        else:
            negative_count += 1.0
        
        for word in current_line:
            # Raw word count
            if word in vocabulary:
                vocabulary[word] += 1.0
            else:
                vocabulary[word] = 1.0
            # Count of word given sentiment
            feature_tup = (word, sentiment)
            if feature_tup in sentiment_dict:
                sentiment_dict[feature_tup] += 1.0
            else:
                sentiment_dict[feature_tup] = 1.0
    
    # Determine most frequent sense from training data
    if positive_count > negative_count:
        mfs_count = positive_count
    else:
        mfs_count = negative_count
    
    sys.stdout.write("Most frequent sense accuracy for training data: ")
    sys.stdout.write(str(mfs_count/corpus_count) + "\n")

    model_dict = calculate_discrimination(vocabulary, sentiment_dict)
    return model_dict


# Use generated model to classify given tweets by sentiment
#    using a decision-list classifier
#
# IN:
#    test_corpus: string of tweets to be analyzed
#    model: dictionary containing sentiment vocabulary
# OUT:
#    output of the function is logged to specified file
def apply_model(test_corpus, model):
    line_count = 0
    test_lines = extract_context(test_corpus)
    for line in test_lines:
        line_count += 1
        current_sentiment = "none"
        current_line = line.split()
        # Decision list with three of my own features
        if "fuck" in current_line:
            current_sentiment = "negative"
        elif "apparently" in current_line:
            current_sentiment = "negative"
        elif "best" in current_line:
            current_sentiment = "positive"
        else:
            # If hand-picked features fail, check sentiment model
            for word in current_line:
                if word in model:
                    current_sentiment = str(model[word][1])
                    break
        
        # If no sentiment was determined, default to "positive"
        if current_sentiment == "none":
            current_sentiment = "positive"

        # Write result to standard output
        sys.stdout.write('sentiment="' + current_sentiment + '"\n')


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
# OUT:
#    sorted_likelihood: dictionary containing sentiment value and
#         log-likelihood, sorted by overall level of discrimination
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

        # print(positive_count)
        positive_prob = float(positive_count)/float(vocab[word])
        negative_prob = float(negative_count)/float(vocab[word])

        discriminator = math.log(positive_prob/negative_prob)

        if (discriminator < 0):
            likelihood_dict[word] = (math.fabs(discriminator), "negative")
        else:
            likelihood_dict[word] = (math.fabs(discriminator), "positive")

        # likelihood_dict[word] = (discriminator)

    # Sort dictionary by total level of discrimination
    sorted_likelihood = dict(sorted(likelihood_dict.items(),
                                    key=lambda x: x[1][0],
                                    reverse=True))

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
    with open(file_path, "w") as f:
        for word in model:
            model_tup = model[word]
            tup_likelihood = model_tup[0]
            tup_sentiment = model_tup[1]

            f.write("Word: " + word + "; Sentiment: " + tup_sentiment +
                    "; Likelihood: " + str(tup_likelihood) + "\n")


if __name__ == "__main__":
    main()
