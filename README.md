# Email Classification with SVM
this project covers data partitioning, ham and spam email classification with support vector machines (SVM),
hyperparameter tuning and k fold cross validation.

## Authors
- [@xingvoong](https://github.com/xingvoong)

## Data
To download the data for this project, visit [this google drive folder](https://drive.google.com/drive/folders/1iknXeSFmPjvSfySFEIZFnFXIvziYyLF2?usp=sharing)


Data is in `.mat` file.  Each `.mat` files will load as a Python dictionary.  Each dictionary contains three fields:
- **training_data**, the training set features. Rows are samples and columns are features.
- **training labels**, the training set labels. Rows are samples. There is one column: The label for each sample.
- **test data**, the test set features. Rows are samples and columns are features.

There are three datasets for this project:
- **mnist_data.mat:** contains data from the MNIST dataset. There are 60,000 labeled digit images for training and 10,000 digit images for testing. The images are grayscale, 28x28 pixels flattened. There are 10 possible labels for each image, namely, the digits 0–9.
example of mnist_data:

![mnist_data](https://raw.githubusercontent.com/xingvoong/email-classification-with-SVM/main/demo/mnist-dataset.png?token=AHX47R57ZN5JQ4GAILKZZIDBME6CS)

- **spam data.mat**: contains featurized spam data. The labels are 1 for spam and 0 for ham
- **cifar10 data.mat**: contains data from the CIFAR10 dataset. There are 50,000 labeled object images for training, and 10,000 object images for testing. The images are flattened 3x32x32 (3 color channels). The labels 0–9 correspond alphabetically to the categories. For example, 0 means airplane, 1 means automobile, 2 means bird, and so on.
example of cifar10_data:

![cifar10](https://raw.githubusercontent.com/xingvoong/email-classification-with-SVM/main/demo/cifar10-dataset.png?token=AHX47R3L6MWC2Z5ZXFISTR3BME6FS)

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

### Classification Accuracy
the code are in `mnist_classify.py`, `spam_classify.py`, `cifar10_classify.py`
- **mnist**: 
```bash
Accuracy 100 0.7764
Accuracy 200 0.8196
Accuracy 500 0.8608
Accuracy 1000 0.8872
Accuracy 2000 0.895
Accuracy 5000 0.9061
Accuracy 10000 0.9094

Error rate [0.22360000000000002, 0.1804, 0.1392,
0.11280000000000001, 0.10499999999999998, 0.09389999999999998,
0.09060000000000001]
```
![mnist_error_rate_graph](https://raw.githubusercontent.com/xingvoong/email-classification-with-SVM/main/demo/mnist_error_rate_graph.png?token=AHX47RYF4BC44CRDZXLNKT3BME5SA)
- **spam**:
```
Accuracy 100 0.7678916827852998
Accuracy 200 0.7882011605415861
Accuracy 500 0.7940038684719536
Accuracy 1000 0.7969052224371374
Accuracy 2000 0.7998065764023211
Accuracy 4138 0.7998065764023211

Error rate [0.23210831721470015,
0.21179883945841393, 
0.2059961315280464, 
0.20309477756286265, 
0.2001934235976789, 
0.2001934235976789]
```
![spam_error_rate_graph](https://raw.githubusercontent.com/xingvoong/email-classification-with-SVM/main/demo/spam_error_rate_graph.png?token=AHX47R7RGH5PUT5IIHTWRILBME5VO)

- **cifar10**:
```
Accuracy 100 0.2364
Accuracy 200 0.2536
Accuracy 500 0.2836
Accuracy 1000 0.2906
Accuracy 2000 0.3202
Accuracy 5000 0.3094

Error rate [0.7636000000000001, 
0.7464, 
0.7163999999999999, 
0.7094, 
0.6798, 
0.6906]
```
![cifar10_error_rate_graph](https://raw.githubusercontent.com/xingvoong/email-classification-with-SVM/main/demo/cifar10_error_rate_graph.png?token=AHX47R76FG53AXZHATGYAEDBME5YA)
## Hyperparameter Tuning
code: `hyperparam_tuning.py` 

- To improve the accuracy, I explored hyperparameter tuning with the regularization parameter or C value, on the mnist dataset.
- To choose a hyperparameter value, I trained the model repeatedly with different hyperparameters, range from 10 to power of -9 to 9.
- The best accuracy occurred at 1e-06, with 93.41%.
- Before generating predictions for the test set,  I retrained the model using all the labeled data and the newly determined hyperparameter

## K-Fold Cross-Validation
code: `k_fold_cross_validation.py`
- Since the spam dataset is relatively small dataset. I used k-fold cross-validation, with k = 5 for spam dataset to improve performance.
- Achieved the best accuracy of 82.42% at C = 100.

## Performance Summary
For three datasets:
- Fitted the model and then used it to predict the validation set.  
- Used a classification accuracy score to calculate accuracy and error rate.  
    - **mnist dataset:** the best accuracy is 93.41%, achieved at C value equals 1e-06
    - **spam**: the best accuracy is 83.49% accuracy, at C = 100
    - **cifar10**: the best accuracy is 39.43%, at C = 10⁸. 

## Run Locally
Clone the project
```bash
https://github.com/xingvoong/email-classification-with-SVM
```
Go to the project directory
```
cd email-classification-with-SVM
```

The project runs in python3.  To check for python version
```bash
python --version
```

Install dependencies
```bash
pip install scikit-learn scipy numpy matplotlib
```

## Requirements
- Python 3
- Linux, macOS, or Windows
