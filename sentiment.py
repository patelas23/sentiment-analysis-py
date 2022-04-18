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
# IN: tweets to be analyzed
# OUT: Trained sentiment classifier
import sys
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

def clean_text():
    pass

def count_features():
    pass

if __name__ == "__main__":
    main()
