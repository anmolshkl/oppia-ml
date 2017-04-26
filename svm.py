from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn import svm
import numpy as np
import os
import random
import time
import yaml

import load_data
import utilities


def main():
    start_time = time.time()
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
    data_load_time = time.time() - start_time

    # Not optimized
    C = 1.0
    classifier_dict = {
        "SVC with linear kernel": svm.SVC(kernel='linear', C=C),
        "SVC with RBF kernel": svm.SVC(kernel='rbf', gamma=0.7, C=C),
        "SVC with polynomial (degree 3) kernel": svm.SVC(kernel='poly', degree=3, C=C),
        "LinearSVC (linear kernel)": svm.LinearSVC(C=C)
    }

    for key, clf in classifier_dict.iteritems():
        start_time = time.time()
        clf.fit(X_train_dtm, y_train)
        y_pred_class = clf.predict(X_test_dtm)
        end_time = time.time()

        print key
        # utilities.print_misclassified_samples(X_test, y_pred_class, y_test)
        utilities.print_stats(y_pred_class, y_test)
        print "Execution time={0} sec \n".format(end_time - start_time + data_load_time)



if __name__ == '__main__':
    # since we are running different classifiers sequentially,
    # using timeit is not possible here
    main()