# Oppia-ml

Oppia is an online learning tool that enables anyone to easily create and share interactive activities (called 'explorations'). These activities simulate a one-on-one conversation with a tutor, making it possible for students to learn by doing and get feedback.

Oppia is written using Python and AngularJS, and is built on top of Google App Engine.

___

This repository is for experimenting ML code for [Oppia](https://github.com/oppia/oppia). It has no dependency with the existing Oppia project and you need not clone the Oppia repository. The existing code in this repository might not follow the same conventions as Oppia's codebase. However, it is still recommended to use the similar coding conventions and style guide, so, feel free to refactor the code as you see fit. Also, it is highly recommended that you use a linux based OS for running the code.

## Dependencies:
Python, NumPy, SciPy, Sci-kit

## Installation:

1. Create a new, empty folder called opensource/ within your home folder. Navigate to it (cd opensource), then [fork and clone](https://help.github.com/articles/fork-a-repo/) the Oppia-ml repo. This will create a new folder named opensource/oppia-ml. Rename this to oppia_ml.

2. Navigate to opensource/oppia_ml/ and run:

  ```
  pip install -r requirements.txt
  ```

## Note:
1. `load_data.load_huge_data()` depends on a json file containing 600,000 tweets. You can download the same from [here](https://drive.google.com/open?id=0B5NwjTjzPp9lZHlHNGdvdnpUQWs).
2. `naive_bayes.py` is custom implementation of Naive Bayes algorithm. 
3. A comparison of different algorithms is available [here](https://docs.google.com/spreadsheets/d/1lWpSh7fuz8xi2oDaHXS_uMtxBqOONquKZPajOYlHl64/edit?usp=sharing).
4. Use `scikit_stress_test.py` for checking the execution time of any particular sci-kit algorithm with datasets of different sizes.
5. `load_data.py` shuffles and divides data into a training set and a testing set. Thus, the performance metrics might differ across different runs of any algorithm.

## Tasks:
1. Use TF-IDF values in naive_bayes.py and check the performance improvement.
2. Implement SVM (linear kernel) and compare its performance with scikit version.
3. Calculate stats for [LDAStringClassifier](https://github.com/oppia/oppia/blob/develop/extensions/classifiers/LDAStringClassifier/LDAStringClassifier.py). [DONE]
4. Use Numpy in naive_bayes.py.

## Results

The dataset comprises of random tweets and hence the vocabulary size is "expected" to vary a lot across the samples. However, we do not expect the same for the answers in Oppia, and, these results should
only be used fo reference purpose. FYI, the models were generated using decision tree classifier.

| Dataset size | Classifier model size (in KB) |
| ------------- | ------------- |
| 69  | 11 |
| 349  | 24 |
| 2500 | 107 |
| 8749 | 618 |
| 22749 | 1800 |
| 34999 | 3300 |
| 43749 | 4100 |

Following are the sizes for different models generated using LinearSVC classifier:

| Dataset size | Classifier model size (in KB) |
| ------------- | ------------- |
| 70  | 6.2 |
| 200  | 14 |
| 400 | 24 |
| 800 | 48 |
| 1700 | 82 |
| 3400 | 170 |
| 7000 | 476 |
| 14000| 846 |
| 28000| 1400 |
| 42000 | 2100 |

The dataset size varies linearly wih the size of the classifier (assumption: the data is really random). For easy visualization, we have plotted the classifier size vs number of training examples and fitted a line:

![visualization](https://github.com/anmolshkl/jusPaid/raw/master/visualization.png)
