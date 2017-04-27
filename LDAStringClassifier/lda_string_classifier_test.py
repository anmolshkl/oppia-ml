import sys
import os.path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn import metrics

import lda_string_classifier
import load_data
import time
import utilities

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_data.load_data()
    classifier = lda_string_classifier.LDAStringClassifier()
    Y_train = [[e] for e in Y_train]
    start_time = time.time()
    classifier.train(zip(X_train, Y_train))
    y_pred_class = classifier.predict(Y_test)
    print "Execution time={0}".format(time.time() - start_time)
    utilities.print_stats(y_pred_class, Y_test)
