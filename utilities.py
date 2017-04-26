from sklearn import metrics


def print_misclassified_samples(X_test, y_pred_class, y_test):
    print "Text || Predicted Class || Actual Class"
    list_of_incorrect = []
    for i in xrange(len(X_test)):
        if y_pred_class[i]!=y_test[i]:
            print "{0} || {1} || {2}|".format(X_test[i], y_pred_class[i], y_test[i])


def print_stats(y_pred_class, y_test):
    print "Accuracy={0}".format(metrics.accuracy_score(y_test, y_pred_class))
    print metrics.classification_report(y_test, y_pred_class)
    print "Macro F1 Score={0}".format(metrics.f1_score(y_test, y_pred_class, average="macro"))
    print "Micro F1 Score={0}".format(metrics.f1_score(y_test, y_pred_class, average="micro"))
    print "Weighted F1 Score={0}".format(metrics.f1_score(y_test, y_pred_class, average="weighted"))
    print "Class-wise F1 Score={0}".format(metrics.f1_score(y_test, y_pred_class, average=None))
