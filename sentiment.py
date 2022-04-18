# sentiment.py 
# Author: Ankur Patel
# Course: Natural Language Processing
# CourseID: CMSC416-SP2022
# Instructor: Caroline Budwell
#
# A decision-list based sentiment classifier for
# identifying sentiment from a given sample of tweets
#
# Usage: python3 sentiment.py training_file.txt test_file.txt model_file.txt > output_answers.txt
#
# IN: training_file: set of tweets annotated with sentiment classification
# 	  test_file: set of tweets to be analyzed for sentiment
#     model_file: name of file to contain sentiment statistics
# OUT: Trained sentiment classifier

import sys
import re

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

def train_model():
    pass

def apply_model():
    pass

def clean_text(corpus_text):
    re.sub(r'', "", corpus_text)

def count_features(feature, context_line):
    pass

# Helper function returns a list of lines from corpus
#     demarcated by:
#         <context> This is a tweet. </context>
def extract_context(corpus_text):
    pass

# Helper function returns a list of sentiment tags
#     corresponding to lines of context
def extract_sentiment(corpus_text):
    return re.findall(r'senseid="(\S+)/>"', corpus_text)


if __name__ == "__main__":
    main()
