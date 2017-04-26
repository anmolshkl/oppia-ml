from io import open
import numpy as np
import os
import random
import time
import yaml


def strip_quotations_newline(text):
    ''' This function is needed when reading tweets from Labeled_Tweets.json
    '''
    text = text.rstrip()
    if text[0] == '"':
        text = text[1:]
    if text[-1] == '"':
        text = text[:-1]
    return text


def split_text(text):
    text = strip_quotations_newline(text)
    splitted_text = text.split(" ")
    cleaned_text = [x for x in splitted_text if len(x)>1]
    text_lowercase = [x.lower() for x in cleaned_text]
    return text_lowercase


def split_dataset(X, Y):
    split_ratio = 0.7
    split_index = int(0.7 * len(X))
    return X[:split_index], Y[:split_index], X[split_index:], Y[split_index:]


def load_data(file=None):
    ''' Loads the training data from a yaml file and returns - input training
    data, output training data, input test data and output test data
    ''' 
    start_time = time.time()
    if file is None:
        file = 'string_classifier_test.yaml'
    yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        file)
    X, Y = [], []
    with open(yaml_path, 'r') as yaml_file:
        yaml_dict = yaml.load(yaml_file)
        interactions = yaml_dict['states']['Home']['interaction']
        # The first element contains no training data,
        # so only consider [1:].
        for answer_group in interactions['answer_groups'][1:]:
            label = answer_group['outcome']['feedback'][0]
            for rule in answer_group['rule_specs']:
                if 'inputs' in rule and 'training_data' in rule['inputs']:
                    for answer in rule['inputs']['training_data']:
                        X.append(answer)
                        Y.append(label)
    combined = list(zip(X, Y))
    random.shuffle(combined)
    X[:], Y[:] = zip(*combined)
    end_time = time.time()
    print "Data load time={0} sec, Please add this up time to the exectuion time".format(end_time - start_time)
    return split_dataset(X, Y)


def load_huge_data(samples=10000):
    ''' Loads `sample` number of examples (around 0.6 million tweets in total).
    Useful for evaluating how much time each classifier takes.
    '''
    start_time = time.time()
    json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Labeled_Tweets.json')
    X, Y = [], []
    with open(json_path, 'r') as json_file:
        lines_gen = islice(json_file, samples)
        for line in lines_gen:
            # Bit of a hack, json_loads wasn't working due to some reason
            tweet = line.split(',')[4].split(':')[1][2:-1]
            X.append(tweet)
            Y.append(random.random() > 0.5)
    combined = list(zip(X[1:100000], Y[1:100000]))
    a = time.time()
    random.shuffle(combined)
    random.shuffle(combined)
    X[:], Y[:] = zip(*combined)
    end_time = time.time()
    print "Data load time={0} sec, Please add this up time to the exectuion time".format(end_time - start_time)
    return split_dataset(X, Y)

if __name__ == "__main__":
        print load_data()
