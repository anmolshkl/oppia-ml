# Oppia-ml

Oppia is an online learning tool that enables anyone to easily create and share interactive activities (called 'explorations'). These activities simulate a one-on-one conversation with a tutor, making it possible for students to learn by doing and get feedback.

Oppia is written using Python and AngularJS, and is built on top of Google App Engine.

___

This repository is for experimenting ML code for [Oppia](https://github.com/oppia/oppia). It has no dependency with the existing Oppia project and you need not clone the Oppia repository. The existing code in this repository might not follow the same conventions as Oppia's codebase. However, it is still recommended to use the similar coding conventions and style guide, so, feel free to refactor the code as you see fit. Also, it is highly recommended that you use a linux based OS for running the code.

## Dependencies:
Python, NumPy, SciPy, Sci-kit

## Installation:

1. Create a new, empty folder called opensource/ within your home folder. Navigate to it (cd opensource), then [fork and clone](https://help.github.com/articles/fork-a-repo/) the Oppia-ml repo. This will create a new folder named opensource/oppia-ml.

2. Navigate to opensource/oppia-ml/ and run:

  ```
  pip install -r requirements.txt
  ```

## Note:
1. `load_data.load_huge_data()` depends on a json file containing 600,000 tweets. You can download the same from (link to be available shortly).
2. `naive_bayes.py` is custom implementation of Naive Bayes algorithm. 
3. A comparison of different algorithms is available here - 
4. Use `scikit_stress_test.py` for checking the execution time of any particular sci-kit algorithm with datasets of different sizes.
5. `load_data.py` shuffles and divides data into a training set and a testing set. Thus, the performance metrics might differ across different runs of any algorithm.

## Tasks:
1. Use TF-IDF values in naive_bayes.py and check the performance improvement.
2. Implement SVM (linear kernel) and compare its performance with scikit version.
3. Calculate stats for [LDAStringClassifier](https://github.com/oppia/oppia/blob/develop/extensions/classifiers/LDAStringClassifier/LDAStringClassifier.py).
