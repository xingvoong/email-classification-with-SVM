import scipy.io
import random

mnist = scipy.io.loadmat('mnist_data.mat')
spam = scipy.io.loadmat('spam_data.mat')
cifar10 = scipy.io.loadmat('cifar10_data.mat')

# mnist
mnist_training_labels = mnist.get('training_labels')
mnist_training_data = mnist.get('training_data')
mnist_training = list(zip(mnist_training_labels, mnist_training_data))
random.shuffle(mnist_training)
mnist_training_labels, mnist_training_data = zip(*mnist_training)

mnist_vali_set = {"mnist_vali_set_labels": mnist_training_labels[0:10000],
                  "mnist_vali_set_data": mnist_training_data[0:10000]}

mnist_remain = {"mnist_remain_labels": mnist_training_labels[10000:],
                "mnist_remain_data": mnist_training_data[10000:]}

scipy.io.savemat('mnist_vali_set.mat', mnist_vali_set)
scipy.io.savemat('mnist_remain.mat', mnist_remain)

# spam
spam_training_labels = spam.get('training_labels')
spam_training_data = spam.get('training_data')
spam_training = list(zip(spam_training_labels, spam_training_data))
random.shuffle(spam_training)
spam_training_labels, spam_training_data = zip(*spam_training)

spam_vali_set = {"spam_vali_set_labels":
                  spam_training_labels[0: int(len(spam_training_labels)*0.2)],
                  "spam_vali_set_data":
                  spam_training_data[0: int(len(spam_training_labels)*0.2)]}

spam_remain = {"spam_remain_labels":
                spam_training_labels[int(len(spam_training_labels)*0.2):],
                "spam_remain_data":
                spam_training_data[int(len(spam_training_labels)*0.2):]}

scipy.io.savemat('spam_vali_set.mat', spam_vali_set)
scipy.io.savemat('spam_remain.mat', spam_remain)

# cifar10
cifar10_training_labels = cifar10.get('training_labels')
cifar10_training_data = cifar10.get('training_data')
cifar10_training = list(zip(cifar10_training_labels, cifar10_training_data))
random.shuffle(cifar10_training)
cifar10_training_labels, cifar10_training_data = zip(*cifar10_training)

cifar10_vali_set = {"cifar10_vali_set_labels":
                    cifar10_training_labels[0: 5000],
                    "cifar10_vali_set_data":
                    cifar10_training_data[0: 5000]}

cifar10_remain = {"cifar10_remain_labels": cifar10_training_labels[5000:],
                  "cifar10_remain_data": cifar10_training_data[5000:]}

scipy.io.savemat('cifar10_vali_set.mat', cifar10_vali_set)
scipy.io.savemat('cifar10_remain.mat', cifar10_remain)
