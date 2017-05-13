from collections import Counter, defaultdict
import load_data as ld
import numpy as np
import time
import utilities

class NaiveBayesTextClassifier():
    """"
    When the goal is classifying text, it is better to give the input X in the form of a list of lists containing words.
    X = [
    ['this', 'is', 'a',...],
    (...)
    ]
    Y still is a 1D array / list containing the labels of each entry
    """

    def calculate_relative_occurences(self, list1):
        no_examples = len(list1)
        ro_dict = dict(Counter(list1))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(no_examples)
	return ro_dict

    def get_max_value_key(self, d1):
        values = d1.values()
        keys = d1.keys()
        max_value_index = values.index(max(values))
        max_key = keys[max_value_index]
        return max_key

    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)

    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = []

    def train(self, X, Y):
        self.class_probabilities = self.calculate_relative_occurences(Y)
        self.labels = np.unique(Y)
        self.no_examples = len(Y)
        self.initialize_nb_dict()
        for ii in range(0,len(Y)):
            label = Y[ii]
            self.nb_dict[label] += X[ii]
        #transform the list with all occurences to a dict with relative occurences
        for label in self.labels:
            self.nb_dict[label] = self.calculate_relative_occurences(self.nb_dict[label])

    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            nb_dict_features = self.nb_dict[label]
            for word in X_elem:
                if word in nb_dict_features.keys():
                    relative_word_occurence = nb_dict_features[word]
                    class_probability *= relative_word_occurence
                else:
                    class_probability *= 0
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)

    def predict(self, X):
        self.predicted_Y_values = []
        n = len(X)
        for ii in range(0,n):
            X_elem = X[ii]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)  
        return self.predicted_Y_values


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = ld.load_data()
    start_time = time.time()
    for i in xrange(len(X_train)):
        X_train[i] = X_train[i].split()
    for i in xrange(len(X_test)):
        X_test[i] = X_test[i].split()
    nbc = NaiveBayesTextClassifier()
    nbc.train(X_train, Y_train)
    y_pred_class = nbc.predict(X_test)
    print 'Execution time={0} sec'.format(time.time() - start_time)
    utilities.print_stats(y_pred_class, Y_test)
