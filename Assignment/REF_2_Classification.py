%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from aml_utils import test_case_checker

df = pd.read_csv('diabetes.csv')
df.head()


# Let's generate the split ourselves.
np_random = np.random.RandomState(seed=12345)
rand_unifs = np_random.uniform(0,1,size=df.shape[0])
division_thresh = np.percentile(rand_unifs, 80)
train_indicator = rand_unifs < division_thresh
eval_indicator = rand_unifs >= division_thresh

# Trainign and Evaluation
train_df = df[train_indicator].reset_index(drop=True)
train_features = train_df.loc[:, train_df.columns != 'Outcome'].values # feature df
train_labels = train_df['Outcome'].values  # labels df
train_df.head()

eval_df = df[eval_indicator].reset_index(drop=True)
eval_features = eval_df.loc[:, eval_df.columns != 'Outcome'].values
eval_labels = eval_df['Outcome'].values
eval_df.head()

#########################
#Pre-Processing

#
# Some of the columns exhibit missing values. 
# We will use a Naive Bayes Classifier later that will treat 
# such missing values in a special way. To be specific, 
# for attribute 3 (Diastolic blood pressure), 
# attribute 4 (Triceps skin fold thickness), 
# attribute 6 (Body mass index), and attribute 8 (Age), 
# we should regard a value of 0 as a missing value.

# Therefore, we will be creating the train_featues_with_nans and 
# eval_features_with_nans numpy arrays to be just like their train_features 
# and eval_features counter-parts, but with the zero-values in such columns 
# replaced with nans.

train_df_with_nans = train_df.copy(deep=True)
eval_df_with_nans = eval_df.copy(deep=True)
for col_with_nans in ['BloodPressure', 'SkinThickness', 'BMI', 'Age']:
    train_df_with_nans[col_with_nans] = train_df_with_nans[col_with_nans].replace(0, np.nan)
    eval_df_with_nans[col_with_nans] = eval_df_with_nans[col_with_nans].replace(0, np.nan)
train_features_with_nans = train_df_with_nans.loc[:, train_df_with_nans.columns != 'Outcome'].values
eval_features_with_nans = eval_df_with_nans.loc[:, eval_df_with_nans.columns != 'Outcome'].values

print('Here are the training rows with at least one missing values.')
print('')
print('You can see that such incomplete data points constitute a substantial part of the data.')
print('')
nan_training_data = train_df_with_nans[train_df_with_nans.isna().any(axis=1)]
nan_training_data

# Task 1

# Building a simple Vaive Bayes Classsifier
def log_prior(train_labels):
    
    # your code here
#     raise NotImplementedError
 # Calculate the probability of y=0 and y=1
    p_y0 = np.mean(train_labels == 0)
    p_y1 = np.mean(train_labels == 1)
    
    # Calculate the log probabilities
    log_py = np.log(np.array([p_y0, p_y1])).reshape(2, 1)
    
    assert log_py.shape == (2,1)
    
    return log_py

# Performing sanity checks on your implementation
some_labels = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1])
some_log_py = log_prior(some_labels)
assert np.array_equal(some_log_py.round(3), np.array([[-0.916], [-0.511]]))

# Checking against the pre-computed test database
test_results = test_case_checker(log_prior, task_id=1)
assert test_results['passed'], test_results['message']

# Testing log_prior
log_py = log_prior(train_labels)
log_py

# Task 2
def cc_mean_ignore_missing(train_features, train_labels):
    N, d = train_features.shape
    
    # your code here
#     raise NotImplementedError
# Calculate the mean for each feature conditional on the label being 0 or 1
    mu_y_0 = np.mean(train_features[train_labels == 0], axis=0)
    mu_y_1 = np.mean(train_features[train_labels == 1], axis=0)
    
    # Combine the results into a single matrix of shape (8, 2)
    mu_y = np.column_stack((mu_y_0, mu_y_1))

    
    assert mu_y.shape == (d, 2)
    return mu_y

# Task 2 implementation
# Performing sanity checks on your implementation
some_feats = np.array([[  1. ,  85. ,  66. ,  29. ,   0. ,  26.6,   0.4,  31. ],
                       [  8. , 183. ,  64. ,   0. ,   0. ,  23.3,   0.7,  32. ],
                       [  1. ,  89. ,  66. ,  23. ,  94. ,  28.1,   0.2,  21. ],
                       [  0. , 137. ,  40. ,  35. , 168. ,  43.1,   2.3,  33. ],
                       [  5. , 116. ,  74. ,   0. ,   0. ,  25.6,   0.2,  30. ]])
some_labels = np.array([0, 1, 0, 1, 0])

some_mu_y = cc_mean_ignore_missing(some_feats, some_labels)

assert np.array_equal(some_mu_y.round(2), np.array([[  2.33,   4.  ],
                                                    [ 96.67, 160.  ],
                                                    [ 68.67,  52.  ],
                                                    [ 17.33,  17.5 ],
                                                    [ 31.33,  84.  ],
                                                    [ 26.77,  33.2 ],
                                                    [  0.27,   1.5 ],
                                                    [ 27.33,  32.5 ]]))

# Checking against the pre-computed test database
test_results = test_case_checker(cc_mean_ignore_missing, task_id=2)
assert test_results['passed'], test_results['message']

mu_y = cc_mean_ignore_missing(train_features, train_labels)
print(mu_y)

# Task 3
def cc_std_ignore_missing(train_features, train_labels):
    N, d = train_features.shape
    
    std_y0 = np.std(train_features[train_labels == 0], axis=0)
    std_y1 = np.std(train_features[train_labels == 1], axis=0)
    
    # Combine the results into a single matrix of shape (8, 2)
    sigma_y = np.column_stack((std_y0, std_y1))
    
    assert sigma_y.shape == (d, 2)
    
    return sigma_y


# Task 3 validation
# Performing sanity checks on your implementation
some_feats = np.array([[  1. ,  85. ,  66. ,  29. ,   0. ,  26.6,   0.4,  31. ],
                       [  8. , 183. ,  64. ,   0. ,   0. ,  23.3,   0.7,  32. ],
                       [  1. ,  89. ,  66. ,  23. ,  94. ,  28.1,   0.2,  21. ],
                       [  0. , 137. ,  40. ,  35. , 168. ,  43.1,   2.3,  33. ],
                       [  5. , 116. ,  74. ,   0. ,   0. ,  25.6,   0.2,  30. ]])
some_labels = np.array([0, 1, 0, 1, 0])

some_std_y = cc_std_ignore_missing(some_feats, some_labels)

assert np.array_equal(some_std_y.round(3), np.array([[ 1.886,  4.   ],
                                                     [13.768, 23.   ],
                                                     [ 3.771, 12.   ],
                                                     [12.499, 17.5  ],
                                                     [44.312, 84.   ],
                                                     [ 1.027,  9.9  ],
                                                     [ 0.094,  0.8  ],
                                                     [ 4.497,  0.5  ]]))

# Checking against the pre-computed test database
test_results = test_case_checker(cc_std_ignore_missing, task_id=3)
assert test_results['passed'], test_results['message']

# call std
sigma_y = cc_std_ignore_missing(train_features, train_labels)
print(sigma_y)


# Task 4
def log_prob(train_features, mu_y, sigma_y, log_py):
    N, d = train_features.shape
    
    # your code here
    # Calculate log probabilities for y = 0 and y = 1
    log_p_x_given_y_0 = -0.5 * np.sum(np.log(2 * np.pi * sigma_y[:, 0]**2) + ((train_features - mu_y[:, 0])**2) / (sigma_y[:, 0]**2), axis=1)
    log_p_x_given_y_1 = -0.5 * np.sum(np.log(2 * np.pi * sigma_y[:, 1]**2) + ((train_features - mu_y[:, 1])**2) / (sigma_y[:, 1]**2), axis=1)
    
    # Add the log prior to get log p(x, y)
    log_p_x_y_0 = log_p_x_given_y_0 + log_py[0]
    log_p_x_y_1 = log_p_x_given_y_1 + log_py[1]
    
    # Stack the results into a (N, 2) matrix
    log_p_x_y = np.column_stack((log_p_x_y_0, log_p_x_y_1))

    
    assert log_p_x_y.shape == (N,2)
    return log_p_x_y

# 1.1 Simple Naives Bayes classifier
class NBClassifier():
    def __init__(self, train_features, train_labels):
        self.train_features = train_features
        self.train_labels = train_labels
        self.log_py = log_prior(train_labels)
        self.mu_y = self.get_cc_means()
        self.sigma_y = self.get_cc_std()
        
    def get_cc_means(self):
        mu_y = cc_mean_ignore_missing(self.train_features, self.train_labels)
        return mu_y
    
    def get_cc_std(self):
        sigma_y = cc_std_ignore_missing(self.train_features, self.train_labels)
        return sigma_y
    
    def predict(self, features):
        log_p_x_y = log_prob(features, self.mu_y, self.sigma_y, self.log_py)
        return log_p_x_y.argmax(axis=1)
    

diabetes_classifier = NBClassifier(train_features, train_labels)
train_pred = diabetes_classifier.predict(train_features)
eval_pred = diabetes_classifier.predict(eval_features)

train_acc = (train_pred==train_labels).mean()
eval_acc = (eval_pred==eval_labels).mean()
print(f'The training data accuracy of your trained model is {train_acc}')
print(f'The evaluation data accuracy of your trained model is {eval_acc}')

# 1.2 Running an off-the-shelf implementation of Naive-Bayes For Comparison

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB().fit(train_features, train_labels)
train_pred_sk = gnb.predict(train_features)
eval_pred_sk = gnb.predict(eval_features)
print(f'The training data accuracy of your trained model is {(train_pred_sk == train_labels).mean()}')
print(f'The evaluation data accuracy of your trained model is {(eval_pred_sk == eval_labels).mean()}')

# Part 2 (Building a Naive Bayes Classifier Considering Missing Entries)

# Task 5

#In this part, we will modify some of the parameter inference functions of the Naive Bayes classifier to make it able to ignore the NaN entries when inferring the Gaussian mean and stds.

def cc_mean_consider_missing(train_features_with_nans, train_labels):
    N, d = train_features_with_nans.shape
    
    # your code here
    # Calculate the mean for each feature conditional on the label being 0 or 1, ignoring NaN values
    mu_y_0 = np.nanmean(train_features_with_nans[train_labels == 0], axis=0)
    mu_y_1 = np.nanmean(train_features_with_nans[train_labels == 1], axis=0)
    
    # Combine the results into a single matrix of shape (d, 2)
    mu_y = np.column_stack((mu_y_0, mu_y_1))
#     raise NotImplementedError
    
    assert not np.isnan(mu_y).any()
    assert mu_y.shape == (d, 2)
    return mu_y

# Task 6
def cc_std_consider_missing(train_features_with_nans, train_labels):
    N, d = train_features_with_nans.shape
    
    # your code here
#     raise NotImplementedError
    
    std_y0 = np.nanstd(train_features_with_nans[train_labels == 0], axis=0)
    std_y1 = np.nanstd(train_features_with_nans[train_labels == 1], axis=0)
    
    # Combine the results into a single matrix of shape (8, 2)
    sigma_y = np.column_stack((std_y0, std_y1))
    
    assert not np.isnan(sigma_y).any()
    assert sigma_y.shape == (d, 2)
    
    return sigma_y

# 2.1 Writing the Naive Bayes Classifier With Missing data handling

class NBClassifierWithMissing(NBClassifier):
    def get_cc_means(self):
        mu_y = cc_mean_consider_missing(self.train_features, self.train_labels)
        return mu_y
    
    def get_cc_std(self):
        sigma_y = cc_std_consider_missing(self.train_features, self.train_labels)
        return sigma_y
    
    def predict(self, features):
        preds = []
        for feature in features:
            is_num = np.logical_not(np.isnan(feature))
            mu_y_not_nan = self.mu_y[is_num,:]
            std_y_not_nan = self.sigma_y[is_num,:]
            feats_not_nan = feature[is_num].reshape(1,-1)
            log_p_x_y = log_prob(feats_not_nan, mu_y_not_nan, std_y_not_nan, self.log_py)
            preds.append(log_p_x_y.argmax(axis=1).item())

        return np.array(preds)
    

diabetes_classifier_nans = NBClassifierWithMissing(train_features_with_nans, train_labels)
train_pred = diabetes_classifier_nans.predict(train_features_with_nans)
eval_pred = diabetes_classifier_nans.predict(eval_features_with_nans)

train_acc = (train_pred==train_labels).mean()
eval_acc = (eval_pred==eval_labels).mean()
print(f'The training data accuracy of your trained model is {train_acc}')
print(f'The evaluation data accuracy of your trained model is {eval_acc}')

# 3. Running SVM Light

# In this section, we are going to investigate the support vector machine classification method. We will become familiar with this classification method in week 3. However, in this section, we are just going to observe how this method performs to set the stage for the third week.
# SVMlight (http://svmlight.joachims.org/) is a famous implementation of the SVM classifier.
# SVMLight can be called from a shell terminal, and there is no nice wrapper for it in python3. Therefore:
# We have to export the training data to a special format called svmlight/libsvm. This can be done using scikit-learn.
# We have to run the svm_learn program to learn the model and then store it.
# We have to import the model back to python.


from sklearn.datasets import dump_svmlight_file
dump_svmlight_file(train_features, 2*train_labels-1, 'training_feats.data', 
                   zero_based=False, comment=None, query_id=None, multilabel=False)


#3.2 training SVM Light

!chmod +x ../BasicClassification-lib/svmlight/svm_learn
from subprocess import Popen, PIPE
process = Popen(["../BasicClassification-lib/svmlight/svm_learn", "./training_feats.data", "svm_model.txt"], stdout=PIPE, stderr=PIPE)
stdout, stderr = process.communicate()
print(stdout.decode("utf-8"))

# 3.3 Importing the SVM Model
from svm2weight import get_svmlight_weights
svm_weights, thresh = get_svmlight_weights('svm_model.txt', printOutput=False)

def svmlight_classifier(train_features):
    return (train_features @ svm_weights - thresh).reshape(-1) >= 0.


train_pred = svmlight_classifier(train_features)
eval_pred = svmlight_classifier(eval_features)

train_acc = (train_pred==train_labels).mean()
eval_acc = (eval_pred==eval_labels).mean()
print(f'The training data accuracy of your trained model is {train_acc}')
print(f'The evaluation data accuracy of your trained model is {eval_acc}')


# Cleaning up after our work is done
!rm -rf svm_model.txt training_feats.data