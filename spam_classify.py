import scipy.io
from sklearn import svm
from sklearn import metrics
import random
import matplotlib.pyplot as plt

spam_remain = scipy.io.loadmat('spam_remain.mat')
spam_vali = scipy.io.loadmat('spam_vali_set.mat')

spam_remain_labels = spam_remain.get("spam_remain_labels")
spam_remain_data = spam_remain.get("spam_remain_data")

clf = svm.SVC(kernel='linear')
all_ = len(spam_remain_labels)

examples = [100, 200, 500, 1000, 2000, all_]
error_rate = [0, 0, 0, 0, 0, 0]
count = 0

# error rate graph
for example in examples:
    clf.fit(spam_remain_data[0:example], spam_remain_labels[0:example].ravel())
    # predict the models
    spam_vali_set_data = spam_vali.get("spam_vali_set_data")
    spam_vali_set_labels = spam_vali.get("spam_vali_set_labels")
    spam_predict = clf.predict(spam_vali_set_data)
    accuracy_score = metrics.accuracy_score(spam_vali_set_labels,
                                            spam_predict)
    error_rate[count] = 1 - accuracy_score
    count += 1
    # Accuracy
    print("Accuracy {}".
          format(example),
          accuracy_score)
print("Error rate", error_rate)

# error rate graph
examples = [100, 200, 500, 1000, 2000, 4138]
plt.suptitle("Error rate graph for spam dataset")
plt.xlabel("error rate")
plt.ylabel("number of examples")
plt.plot(error_rate, examples, '.r-')
plt.show()
