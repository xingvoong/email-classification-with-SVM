import scipy.io
import random
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

spam = scipy.io.loadmat('spam_data.mat')

spam_labels = spam.get('training_labels')
spam_data = spam.get('training_data')

spam_training = list(zip(spam_labels, spam_data))
random.shuffle(spam_training)
spam_labels, spam_data = zip(*spam_training)

elements_per_bin = int(len(spam_labels)/5)
vali = [None] * 5
spam_predict = [None] * 5
count = 1

for i in np.arange(1, 6, 1):
    vali[i-1] = {"vali_labels":
                 spam_labels[elements_per_bin*(i-1):
                             elements_per_bin*i],
                 "vali_data":
                 spam_data[elements_per_bin*(i-1):
                           elements_per_bin*i]}

for c in (10**p for p in range(-9, 10)):
    print("C value: ", c)
    count = 1
    total = average = 0.0
    for v in vali:
        clf = svm.LinearSVC(C=c)
        cur_samples = spam_data[(count-1):elements_per_bin*count]
        cur_labels = np.array(
                     spam_labels[(count-1):elements_per_bin*count]).ravel()
        clf.fit(cur_samples, cur_labels)
        spam_predict[count-1] = clf.predict(v.get("vali_data"))
        accuracy = metrics.accuracy_score(v.get("vali_labels"),
                                          spam_predict[count-1])
        print("Accuracy for validation set {}:".format(count), accuracy)
        count += 1
        total += accuracy
    average = total/5
    print("The average accuracy for C = {}: ".format(c), average)
    print("")
