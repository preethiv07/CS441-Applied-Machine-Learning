# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from aml_utils import test_case_checker

df = pd.read_csv('diabetes.csv')
df.head()

def simple_pred_vec(g, theta):
    
    # your code here
    out = (g >= theta)
#     raise NotImplementedError
    
    return out

# Test for simple_pred_vec
#####################
# Performing sanity checks on your implementation
# assert (simple_pred_vec(g=np.array([100., 200., 140.]), theta=140.) == np.array([False, True, True])).all()

# # Checking against the pre-computed test database
# test_results = test_case_checker(simple_pred_vec, task_id=1)
# assert test_results['passed'], test_results['message']
######################

def simple_pred(df, theta):
    
    # your code here
    pred = simple_pred_vec(df['Glucose'].values.reshape(1, -1), theta)
#     raise NotImplementedError
    
    return pred

#####################
# # Performing sanity checks on your implementation
# assert np.array_equal(simple_pred(df, 120)[:,:5], np.array([[True, False,  True, False,  True]]))

# # Checking against the pre-computed test database
# test_results = test_case_checker(simple_pred, task_id=2)
# assert test_results['passed'], test_results['message']
#####################

def simple_acc(df, theta):
    
    if np.isscalar(theta):
        theta = np.array([theta]).reshape(-1, 1)  
    actual_outcomes = df['Outcome'].values.reshape(1, -1)
    acc = np.mean(simple_pred(df, theta) == actual_outcomes, axis=1)
    return acc

# Performing sanity checks on your implementation

#####################
# Toy testing the shapes
# assert simple_acc(df, theta=120).shape == (1,)
# assert simple_acc(df, theta=np.array([50,100,300]).reshape(3,1)).shape == (3,)

# # Toy testing the values
# assert simple_acc(df, theta=120).round(3)==0.698
# assert np.array_equal(simple_acc(df, theta=np.array([[50,100,20,40]]).T).round(3), [0.352, 0.564, 0.35 , 0.35 ])

# # Checking against the pre-computed test database
# test_results = test_case_checker(simple_acc, task_id=3)
# assert test_results['passed'], test_results['message']
#####################

# print(simple_acc(df.iloc[:10, :], theta=120))
# print(simple_acc(df.iloc[:10, :], theta=np.array([50,100,300]).reshape(3,1)))

# print(np.array([50,100,300]).reshape(3,1))