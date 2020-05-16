import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    with open(f"{filename}", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tmp_evidence = []
            tmp_revenue = 0
            for i in row:
                temp = row[i]
                if i == "Month":
                    if temp == "Jan":
                        temp = 0
                    elif temp == "Feb":
                        temp = 1
                    elif temp == "Mar":
                        temp = 2
                    elif temp == "Apr":
                        temp = 3
                    elif temp == "May":
                        temp = 4
                    elif temp == "June":
                        temp = 5
                    elif temp == "Jul":
                        temp = 6
                    elif temp == "Aug":
                        temp = 7
                    elif temp == "Sep":
                        temp = 8
                    elif temp == "Oct":
                        temp = 9
                    elif temp == "Nov":
                        temp = 10
                    elif temp == "Dec":
                        temp = 11
                elif i == "VisitorType":
                    if temp == "Returning_Visitor":
                        temp = 1
                    else:
                        temp = 0
                elif i == "Weekend":
                    if temp == "TRUE":
                        temp = 1
                    elif temp == "FALSE":
                        temp = 0
                elif i == "Revenue":
                    if temp == "TRUE":
                        temp = 1
                    elif temp == "FALSE":
                        temp = 0
                elif i == "Administrative" or i == "Informational" or i == "ProductRelated" or i == "Month" or i == "OperatingSystems" or i == "Browser" or i == "Region" or i == "TrafficType":
                    temp = int(temp)
                else:
                    temp = float(temp)
                if i != "Revenue":
                    tmp_evidence.append(temp)
                else:
                    tmp_revenue = temp
            evidence.append(tmp_evidence)
            labels.append(tmp_revenue)
    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    neighbor = KNeighborsClassifier(n_neighbors=1)
    neighbor.fit(evidence, labels)
    return neighbor


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    buy = 0
    no_buy = 0
    buy_correct = 0
    no_buy_correct = 0
    length = len(labels)
    for i in range(length):
        if labels[i] == 1:
            buy += 1
            if predictions[i] == 1:
                buy_correct += 1
        else:
            no_buy += 1
            if predictions[i] == 0:
                no_buy_correct += 1

    sensitivity = buy_correct/buy
    specificty = no_buy_correct/no_buy

    return (sensitivity, specificty)


if __name__ == "__main__":
    main()
