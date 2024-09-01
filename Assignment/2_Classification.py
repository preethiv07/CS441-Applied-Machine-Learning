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