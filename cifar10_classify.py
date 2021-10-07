
import scipy.io
from sklearn import svm
from sklearn import metrics
import random
import matplotlib.pyplot as plt

cifar10_remain = scipy.io.loadmat('cifar10_remain.mat')
cifar10_vali = scipy.io.loadmat('cifar10_vali_set.mat')

cifar10_remain_labels = cifar10_remain.get("cifar10_remain_labels")
cifar10_remain_data = cifar10_remain.get("cifar10_remain_data")

clf = svm.SVC(kernel='linear')

examples = [100, 200, 500, 1000, 2000, 5000]
error_rate = [0, 0, 0, 0, 0, 0]
count = 0
for example in examples:
    clf.fit(cifar10_remain_data[0:example],
            cifar10_remain_labels[0:example].ravel())
    # predict the models
    cifar10_vali_set_data = cifar10_vali.get("cifar10_vali_set_data")
    cifar10_vali_set_labels = cifar10_vali.get("cifar10_vali_set_labels")
    cifar10_predict = clf.predict(cifar10_vali_set_data)
    accuracy_score = metrics.accuracy_score(cifar10_vali_set_labels,
                                            cifar10_predict)
    error_rate[count] = 1 - accuracy_score
    # Accuracy
    print("Accuracy {}".format(example), accuracy_score)
    count += 1
print("Error rate", error_rate)

# error rate graph
examples = [100, 200, 500, 1000, 2000, 5000]
plt.suptitle("Error rate graph for cifar10 dataset")
plt.xlabel("error rate")
plt.ylabel("number of examples")
plt.plot(error_rate, examples, '.r-')
plt.show()
