import scipy.io
from sklearn import svm
from sklearn import metrics
import random
import matplotlib.pyplot as plt

mnist_remain = scipy.io.loadmat('mnist_remain.mat')
mnist_vali = scipy.io.loadmat('mnist_vali_set.mat')

mnist_remain_labels = mnist_remain.get("mnist_remain_labels")
mnist_remain_data = mnist_remain.get("mnist_remain_data")

for c in (10**p for p in range(-9, 10)):
    clf = svm.SVC(kernel='linear', C=c)
    print("C value: ", c)
    clf.fit(mnist_remain_data[0:10000], mnist_remain_labels[0:10000].ravel())
    mnist_vali_set_data = mnist_vali.get("mnist_vali_set_data")
    mnist_vali_set_labels = mnist_vali.get("mnist_vali_set_labels")
    mnist_predict = clf.p redict(mnist_vali_set_data)
    accuracy_score = metrics.accuracy_score(mnist_vali_set_labels,
                                            mnist_predict)
    print("Accuracy for 10000 examples: ", accuracy_score)
    print(" ")
