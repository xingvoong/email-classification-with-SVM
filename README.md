# Email Classifer with SVM
this project covers data partitioning, ham and spam email classification with support vector machines (SVM),
hyperparameter tuning and k fold cross validation.
## Checklist
- [ ] change repo name to **Email Classifer with SVM**
- [ ] Insert picture for data example for mnist-dataset

## Authors
- [@xingvoong](https://github.com/xingvoong)

## Data
Data is in .mat file.  Each .mat files will load as a Python dictionary.  Each dictionary contains three fields:
- **training_data**, the training set features. Rows are samples and columns are features.
- **training labels**, the training set labels. Rows are samples. There is one column: The label for each sample.
- **test data**, the test set features. Rows are samples and columns are features.  Fits a model to predict the labels for this test set, and submit those predictions to Kaggle.

There are three datasets for this project:
- **mnist_data.mat:** contains data from the MNIST dataset. There are 60,000 labeled digit images for training and 10,000 digit images for testing. The images are grayscale, 28x28 pixels flattened. There are 10 possible labels for each image, namely, the digits 0–9.
(insert picture of data here)
- **spam data.mat**: contains featurized spam data. The labels are 1 for spam and 0 for ham
- **cifar10 data.mat**: contains data from the CIFAR10 dataset. There are 50,000 labeled object images for training, and 10,000 object images for testing. The images are flattened 3x32x32 (3 color channels). The labels 0–9 correspond alphabetically to the categories. For example, 0 means airplane, 1 means automobile, 2 means bird, and so on.

## Data partitioning
the code can be found in `data_partitioning.py`

First, shuffle the data so that all the classes are represented in the partitions.  Then partition given data to "training" data and "validation" data.
- **mnist dataset**: set aside 10,000 training images as a validation set.
- **spam dataset**: 20% of of the training data as a validation set.
- **cifar10 dataset**: sets aside 5,000 training images as a validation set.

## Classifer with SVM
Train a linear support vector machine (SVM) on all three datasets. Plot the error rate on the training
and validation sets versus the number of training examples that are used to train the classifier.
The number of training examples in the experiment varies per dataset.

- **Model evaluation:** classification accuracy for error rate
- **Features**:
    - mnist: raw pixels
    - spam: word frequencies
    - cifar10: raw pixels




## Performance Report