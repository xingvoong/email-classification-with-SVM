import scipy.io
import random
from sklearn import svm
import pandas as pd
import numpy as np

mnist = scipy.io.loadmat('mnist_data.mat')

clf = svm.SVC(kernel='linear', C=1e-06)
clf.fit(mnist.get('training_data'), mnist.get('training_labels').ravel())

# A code snippet to save results into a csv file.
# Usage results_to_csv(clf.predict(X_test))


def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('mnist_result.csv', index_label='Id')

results_to_csv(clf.predict(mnist.get('test_data')))
