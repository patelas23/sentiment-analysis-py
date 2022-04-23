# scorer.py
# Author: Ankur Patel
# Course: Natural Language Processing
# CourseID: CMSC-416-SP2022
# Instructor: Bridgette Mckinnes

# Helper program for generating grading the accuracy of sentiment.py
#     generating a confusion matrix for analysis

# Imports
import sys
import pandas as pd
from sklearn import metrics

def main():
    print("Welcome to sentiment scorer.py!")
    my_answer_file = sys.argv[1]
    answer_key_file = sys.argv[2]

    my_answer_string = ""
    answer_key_string = ""

    with open(my_answer_file) as f:
        my_answer_string = f.read()

    with open(answer_key_file) as f:
        answer_key_string = f.read()

    my_sentiment_answers = get_sentiment(my_answer_string)
    sentiment_answer_key = get_sentiment(answer_key_string)

    get_confusion_stats(sentiment_answer_key, my_sentiment_answers)

# Create map of sense words from answer key

# Extract sentences from test data


# Compare actual and expected values to produce confusion matrix
def get_confusion_stats(actual, predicted):
    actual_series = pd.Series(actual, name="Actual")
    predicted_series = pd.Series(predicted, name="Predicted")
    sys.stdout.write("Accuracy Score: " +
                     str(metrics.accuracy_scoreaccur(actual,
                         predicted + "\n")))
    sys.stdout.write("Confusion Matrix \n" +
                     str(pd.crosstab(actual_series, predicted_series)))




if __name__ == "__main__":
    main()
