import scipy.io
from sklearn import svm
from sklearn import metrics
import random
import matplotlib.pyplot as plt

mnist_remain = scipy.io.loadmat('mnist_remain.mat')
mnist_vali = scipy.io.loadmat('mnist_vali_set.mat')

mnist_remain_labels = mnist_remain.get("mnist_remain_labels")
mnist_remain_data = mnist_remain.get("mnist_remain_data")

clf = svm.SVC(kernel='linear')

examples = [100, 200, 500, 1000, 2000, 5000, 10000]
error_rate = [0, 0, 0, 0, 0, 0, 0]
count = 0
for example in examples:
    cur_remain_data = mnist_remain_data[0:example]
    cur_remain_labels = mnist_remain_labels[0:example].ravel()
    clf.fit(cur_remain_data, cur_remain_labels)
    # predict the models
    mnist_vali_set_data = mnist_vali.get("mnist_vali_set_data")
    mnist_vali_set_labels = mnist_vali.get("mnist_vali_set_labels")
    mnist_predict = clf.predict(mnist_vali_set_data)
    accuracy_score = metrics.accuracy_score(mnist_vali_set_labels, mnist_predict)
    error_rate[count] = 1 - accuracy_score
    # Accuracy
    print("Accuracy {}".format(example),
          metrics.accuracy_score(mnist_vali_set_labels, mnist_predict))
    count += 1

print("Error rate", error_rate)

# error rate graph
examples = [100, 200, 500, 1000, 2000, 5000, 10000]
error_rate = [0.2671,
              0.19589999999999996,
              0.13819999999999999,
              0.11839999999999995,
              0.10970000000000002,
              0.098300000000000054,
              0.08879999999999999]
plt.suptitle("Error rate graph for MNIST dataset")
plt.xlabel("error rate")
plt.ylabel("number of examples")
plt.plot(error_rate, examples, '.r-')
plt.show()
