from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
import numpy as np
import os
import random
import time
import timeit
import yaml

import load_data
import utilities


def main():
    # Load data
    X_train, y_train, X_test, y_test = load_data.load_data()
    # Transform data into a vector of TF-IDF values
    count_vect = CountVectorizer(ngram_range=(1, 2))
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer(use_idf=True)
    X_train_dtm = tfidf_transformer.fit_transform(X_train_counts)
    # Transform test data
    X_test_counts = count_vect.transform(X_test)
    X_test_dtm = tfidf_transformer.fit_transform(X_test_counts)

    # Not optimized, probably need to test with l2 penalty also
    clf = LogisticRegression(penalty='l1')
    clf.fit(X_train_dtm, y_train)
    y_pred_class = clf.predict(X_test_dtm)

    # utilities.print_misclassified_samples(X_test, y_pred_class, y_test)
    utilities.print_stats(y_pred_class, y_test)


if __name__ == '__main__':
    # probably not the best way to measure time, but, we only want a ballpark figure
    execution_time = timeit.timeit("main()", setup="from __main__ import main", number=1)
    print "Execution time={0} sec".format(execution_time)