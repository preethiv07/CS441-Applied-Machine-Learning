%matplotlib inline
%load_ext autoreload
%autoreload 2

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
from glmnet import glmnet; from glmnetPlot import glmnetPlot 
from glmnetPrint import glmnetPrint; from glmnetCoef import glmnetCoef; from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot; from cvglmnetPredict import cvglmnetPredict

from aml_utils import test_case_checker, perform_computation
# Note: AML_Utils are unknown teset checker

warnings.filterwarnings('ignore')

df = pd.read_csv('../GLMnet-lib/music/default_plus_chromatic_features_1059_tracks.txt', header=None)
df

#Information Summary
# Input/Output: This data has 118 columns; the first 116 columns are the music features, and the last two columns are the music origin's latitude and the longitude, respectively.
# Missing Data: There is no missing data.
# Final Goal: We want to properly fit a linear regression model.

X_full = df.iloc[:,:-2].values
lat_full = df.iloc[:,-2].values
lon_full = df.iloc[:,-1].values
X_full.shape, lat_full.shape, lon_full.shape

# Output: ((1059, 116), (1059,), (1059,))

# Making the Dependent Variables Positive
<!-- This will make the data compatible with the box-cox transformation that we will later use. -->

lat_full = 90 + lat_full
lon_full = 180 + lon_full

# Outlier detection

outlier_detector = 'LOF'

if outlier_detector == 'LOF':
    outlier_clf = LocalOutlierFactor(novelty=False)
elif outlier_detector == 'IF':
    outlier_clf = IsolationForest(warm_start=True, random_state=12345)
elif outlier_detector == 'EE':
    outlier_clf = EllipticEnvelope(random_state=12345)
else:
    outlier_clf = None

is_not_outlier = outlier_clf.fit_predict(X_full) if outlier_clf is not None else np.ones_like(lat_full)>0
X_useful = X_full[is_not_outlier==1,:]
lat_useful = lat_full[is_not_outlier==1]
lon_useful = lon_full[is_not_outlier==1]

##########
# 1.2 Train-validation-test-split
##########
train_val_indices, test_indices = train_test_split(np.arange(X_useful.shape[0]), test_size=0.2, random_state=12345)

X_train_val = X_useful[train_val_indices, :]
lat_train_val = lat_useful[train_val_indices]
lon_train_val = lon_useful[train_val_indices]

X_test = X_useful[test_indices, :]
lat_test = lat_useful[test_indices]
lon_test = lon_useful[test_indices]

##########
# # 1.3 building simple regression
##########
from sklearn.linear_model import LinearRegression

if perform_computation:
    X, Y = X_train_val, lat_train_val
    reg_lat = LinearRegression().fit(X, Y)
    train_r2_lat = reg_lat.score(X,Y)
    fitted_lat = reg_lat.predict(X)
    residuals_lat = Y-fitted_lat
    train_mse_lat = (residuals_lat**2).mean()
    test_mse_lat = np.mean((reg_lat.predict(X_test)-lat_test)**2)
    test_r2_lat = reg_lat.score(X_test,lat_test)

    X, Y = X_train_val, lon_train_val
    reg_lon = LinearRegression().fit(X, Y)
    train_r2_lon = reg_lon.score(X,Y)
    fitted_lon = reg_lon.predict(X)
    residuals_lon = Y-fitted_lon
    train_mse_lon = (residuals_lon**2).mean()
    test_mse_lon = np.mean((reg_lon.predict(X_test)-lon_test)**2)
    test_r2_lon = reg_lon.score(X_test,lon_test)

    fig, axes = plt.subplots(1,2, figsize=(10,6.), dpi=100)

    ax = axes[0]
    ax.scatter(fitted_lat, residuals_lat)
    ax.set_xlabel('Fitted Latitude')
    ax.set_ylabel('Latitude Residuals')
    _ = ax.set_title(f'Residuals Vs. Fitted Latitude.\n' +
                     f'Training R2=%.3f, Testing R2=%.3f\n' % (train_r2_lat, test_r2_lat) +
                     f'Training MSE=%.3f, Testing MSE=%.3f' % (train_mse_lat, test_mse_lat))

    ax = axes[1]
    ax.scatter(fitted_lon, residuals_lon)
    ax.set_xlabel('Fitted Longitude')
    ax.set_ylabel('Longitude Residuals')
    _ = ax.set_title(f'Residuals Vs. Fitted Longitude.\n' +
                     f'Training R2=%.3f, Testing R2=%.3f\n' % (train_r2_lon, test_r2_lon) +
                     f'Training MSE=%.3f, Testing MSE=%.3f' % (train_mse_lon, test_mse_lon))
    fig.set_tight_layout([0, 0, 1, 1])

# # 1.4
# Write a function glmnet_vanilla that fits a linear regression model from the glmnet library, and takes the following arguments as input:
# X_train: A numpy array of the shape (N,d) where N is the number of training data points, and d is the data dimension. Do not assume anything about N or d other than being a positive integer.
# Y_train: A numpy array of the shape (N,) where N is the number of training data points.
# X_test: A numpy array of the shape (N_test,d) where N_test is the number of testing data points, and d is the data dimension.
# Your model should train on the training features and labels, and then predict on the test data. Your model should return the following two items:
# fitted_Y: The predicted values on the test data as a numpy array with a shape of (N_test,) where N_test is the number of testing data points.
# glmnet_model: The glmnet library's returned model stored as a python dictionary.

def glmnet_vanilla(X_train, Y_train, X_test=None):
    """
    Train a linear regression model using the glmnet library.
    
        Parameters:
                X_train (np.array): A numpy array of the shape (N,d) where N is the number of training data points, and d is the data dimension. 
                Y_train (np.array): A numpy array of the shape (N,) where N is the number of training data points.
                X_test (np.array): A numpy array of the shape (N_test,d) where N_test is the number of testing data points, and d is the data dimension.
                
        Returns:
                fitted_Y (np.array): The predicted values on the test data as a numpy array with a shape of (N_test,) where N_test is the number of testing data points.
                glmneet_model (dict): The glmnet library's returned model stored as a python dictionary.
    """
    if X_test is None:
        X_test = X_train.copy().astype(np.float64)
    # Creating Scratch Variables For glmnet Consumption
    X_train = X_train.copy().astype(np.float64)
    Y_train = Y_train.copy().astype(np.float64)
    
    # your code here
     # Train the model using the 'gaussian' family for linear regression
    glmnet_model = glmnet(x=X_train, y=Y_train, family = 'gaussian')
    
    # Perform prediction on the test data with s=lambda=array of zero (no regularization)
    fitted_Y = glmnetPredict(glmnet_model, X_test, ptype = 'response', s = scipy.float64([0])).reshape((X_test.shape[0],))
    
    
    assert fitted_Y.shape == (X_test.shape[0],), 'fitted_Y should not be two dimensional (hint: reshaping may be helpful)'
    assert isinstance(glmnet_model, dict)
    assert list(glmnet_model.keys()) == ['a0','beta','dev','nulldev','df','lambdau','npasses','jerr','dim','offset','class']
    return fitted_Y, glmnet_model

#Given Code
def train_and_plot(trainer):
    # Latitude Training, Prediction, Evaluation, etc.
    lat_pred_train = trainer(X_train_val, lat_train_val, X_train_val)[0]
    train_r2_lat = r2_score(lat_train_val, lat_pred_train)
    residuals_lat = lat_train_val - lat_pred_train
    train_mse_lat = (residuals_lat**2).mean()
    lat_pred_test = trainer(X_train_val, lat_train_val, X_test)[0]
    test_mse_lat = np.mean((lat_pred_test-lat_test)**2)
    test_r2_lat = r2_score(lat_test, lat_pred_test)

    # Longitude Training, Prediction, Evaluation, etc.
    lon_pred_train = trainer(X_train_val, lon_train_val, X_train_val)[0]
    train_r2_lon = r2_score(lon_train_val, lon_pred_train)
    residuals_lon = lon_train_val - lon_pred_train
    train_mse_lon = (residuals_lon**2).mean()
    lon_pred_test = trainer(X_train_val, lon_train_val, X_test)[0]
    test_mse_lon = np.mean((lon_pred_test-lon_test)**2)
    test_r2_lon = r2_score(lon_test, lon_pred_test)

    fig, axes = plt.subplots(1,2, figsize=(10,6.), dpi=100)

    ax = axes[0]
    ax.scatter(lat_pred_train, residuals_lat)
    ax.set_xlabel('Fitted Latitude')
    ax.set_ylabel('Latitude Residuals')
    _ = ax.set_title(f'Residuals Vs. Fitted Latitude.\n' +
                     f'Training R2=%.3f, Testing R2=%.3f\n' % (train_r2_lat, test_r2_lat) +
                     f'Training MSE=%.3f, Testing MSE=%.3f' % (train_mse_lat, test_mse_lat))

    ax = axes[1]
    ax.scatter(lon_pred_train, residuals_lon)
    ax.set_xlabel('Fitted Longitude')
    ax.set_ylabel('Longitude Residuals')
    _ = ax.set_title(f'Residuals Vs. Fitted Longitude.\n' +
                     f'Training R2=%.3f, Testing R2=%.3f\n' % (train_r2_lon, test_r2_lon) +
                     f'Training MSE=%.3f, Testing MSE=%.3f' % (train_mse_lon, test_mse_lon))
    fig.set_tight_layout([0, 0, 1, 1])
    
if perform_computation:
    train_and_plot(glmnet_vanilla)

######
# Bocx- Cox
######
def boxcox_lambda(y):
    """
    Find the best box-cox transformation 𝜆 parameter `best_lam` as a scalar.
    
        Parameters:
                y (np.array): A numpy array
                
        Returns:
                best_lam (np.float64): The best box-cox transformation 𝜆 parameter
    """    
    assert y.ndim==1
    assert (y>0).all()
    
    # your code here

    
    return best_lam

######
# Task 3: Task 3
# Write a function boxcox_transform that takes a numpy array y and the box-cox transformation
#  𝜆 parameter lam as input, and returns the numpy array transformed_y which is the box-cox transformation of y using 𝜆.
#####

def boxcox_transform(y, lam):
    """
    Perform the box-cox transformation over array y using 𝜆
    
        Parameters:
                y (np.array): A numpy array
                lam (np.float64): The box-cox transformation 𝜆 parameter
                
        Returns:
                transformed_y (np.array): The numpy array after box-cox transformation using 𝜆
    """
    assert y.ndim==1
    assert (y>0).all()
    
    # your code here
    transformed_y = scipy.stats.boxcox(y, lam)

    
    return transformed_y

###### Task 4#####
# Write a function boxcox_inv_transform that takes a numpy array transformed_y a
# nd the box-cox transformation  𝜆  parameter lam as input, and returns the numpy array y 
# which is the inverse box-cox transformation of transformed_y using  𝜆 .
# If  𝜆≠0 :𝑦=|𝑦^𝑏𝑐⋅𝜆+1|^1/𝜆
# If  𝜆=0 :𝑦=𝑒^𝑦𝑏𝑐
# Hint: You need to implement this function yourself!
# Important Note: Be very careful about the signs, absolute values, and raising to exponents 
# with decimal points. For something to be raised to any power that is not a full integer, 
# you need to make sure that the base is positive.

def boxcox_inv_transform(transformed_y, lam):
    """
    Perform the invserse box-cox transformation over transformed_y using 𝜆
    
        Parameters:
                transformed_y (np.array): A numpy array after box-cox transformation
                lam (np.float64): The box-cox transformation 𝜆 parameter
                
        Returns:
                y (np.array): The numpy array before box-cox transformation using 𝜆
    """
    
    # your code here
    # Case 1: When lambda is not zero
    if lam != 0:
        y = np.power(np.abs(transformed_y * lam + 1), (1 / lam))
    # Case 2: When lambda is zero
    else:
        y = np.exp(transformed_y)
    
    assert not np.isnan(y).any()
    return y

######## TASK 5 ########
# Using the box-cox functions you previously wrote, write a function glmnet_bc that fits a linear regression model
# from the glmnet library with the box-cox transformation applied on the labels, and takes the following arguments as input:

def glmnet_bc(X_train, Y_train, X_test=None):
    """
    Train a linear regression model using the glmnet library with the box-cox transformation.
    
        Parameters:
                X_train (np.array): A numpy array of the shape (N,d) where N is the number of training data points, and d is the data dimension. 
                Y_train (np.array): A numpy array of the shape (N,) where N is the number of training data points.
                X_test (np.array): A numpy array of the shape (N_test,d) where N_test is the number of testing data points, and d is the data dimension.
                
        Returns:
                fitted_test (np.array): The predicted values on the test data as a numpy array with a shape of (N_test,) where N_test is the number of testing data points.
                glmneet_model (dict): The glmnet library's returned model stored as a python dictionary.
    """
    # your code here

    # Step 1: Find the best Box-Cox lambda using the training labels
    best_lam = scipy.stats.boxcox_normmax(Y_train, method='mle')
#     best_lam = boxcox_lambda(Y_train)

    # Step 2: Apply the Box-Cox transformation on the training labels
    Y_train_transformed = scipy.stats.boxcox(Y_train, lmbda=best_lam)
    
    # Step 3: Train the model using glmnet_vanilla on the transformed training data
    fitted_transformed_test, glmnet_model = glmnet_vanilla(X_train, Y_train_transformed, X_test)
    
    # Step 4: Inverse Box-Cox transform on the predicted test data
    fitted_test = boxcox_inv_transform(fitted_transformed_test, best_lam)
    
    assert isinstance(glmnet_model, dict)
    return fitted_test, glmnet_model


####### TASK 6 ##########
def glmnet_ridge(X_train, Y_train, X_test=None):
    """
    Train a Ridge-regression model using the glmnet library.
    
        Parameters:
                X_train (np.array): A numpy array of the shape (N,d) where N is the number of training data points, and d is the data dimension. 
                Y_train (np.array): A numpy array of the shape (N,) where N is the number of training data points.
                X_test (np.array): A numpy array of the shape (N_test,d) where N_test is the number of testing data points, and d is the data dimension.
                
        Returns:
                fitted_Y_test (np.array): The predicted values on the test data as a numpy array with a shape of (N_test,) where N_test is the number of testing data points.
                glmneet_model (dict): The glmnet library's returned model stored as a python dictionary.
    """    
    if X_test is None:
        X_test = X_train.copy().astype(np.float64)
    # Creating Scratch Variables For glmnet Consumption
    X_train = X_train.copy().astype(np.float64)
    Y_train = Y_train.copy().astype(np.float64)
    
    # your code here
    glmnet_model = cvglmnet(x=X_train.copy(), y=Y_train.copy(), family ='gaussian', ptype = 'mse', 
                            nfolds = 10, nlambda=100, alpha=0)
    #Note Alpha= 0 it is Lasso, Alpha=1 is Ridge, between 0 and 1 is GLMnet
    
    fitted_Y_test = cvglmnetPredict(glmnet_model, X_test, ptype="response", 
                                    s='lambda_min').reshape((X_test.shape[0],))
    
    
    assert fitted_Y_test.shape == (X_test.shape[0],), 'fitted_Y should not be two dimensional (hint: reshaping may be helpful)'
    assert isinstance(glmnet_model, dict)
    
    return fitted_Y_test, glmnet_model


########## Task 7 Lasso ###########
def glmnet_lasso(X_train, Y_train, X_test=None):
    """
    Train a Lasso-regression model using the glmnet library.
    
        Parameters:
                X_train (np.array): A numpy array of the shape (N,d) where N is the number of training data points, and d is the data dimension. 
                Y_train (np.array): A numpy array of the shape (N,) where N is the number of training data points.
                X_test (np.array): A numpy array of the shape (N_test,d) where N_test is the number of testing data points, and d is the data dimension.
                
        Returns:
                fitted_Y_test (np.array): The predicted values on the test data as a numpy array with a shape of (N_test,) where N_test is the number of testing data points.
                glmneet_model (dict): The glmnet library's returned model stored as a python dictionary.
    """        
    if X_test is None:
        X_test = X_train.copy().astype(np.float64)
    # Creating Scratch Variables For glmnet Consumption
    X_train = X_train.copy().astype(np.float64)
    Y_train = Y_train.copy().astype(np.float64)
    
    # your code here
    glmnet_model = cvglmnet(x=X_train.copy(), y=Y_train.copy(), family ='gaussian', ptype = 'mse', 
                            nfolds = 10, nlambda=100, alpha=1)
    #Note Alpha= 0 it is Lasso, Alpha=1 is Ridge, between 0 and 1 is GLMnet
    
    fitted_Y_test = cvglmnetPredict(glmnet_model, X_test, ptype="response", 
                                    s='lambda_min').reshape((X_test.shape[0],))
    

    
    assert fitted_Y_test.shape == (X_test.shape[0],), 'fitted_Y should not be two dimensional (hint: reshaping may be helpful)'
    assert isinstance(glmnet_model, dict)
    return fitted_Y_test, glmnet_model


if perform_computation:
    _, lasso_model = glmnet_lasso(X_train_val, lat_train_val, X_train_val)
    _, ridge_model = glmnet_ridge(X_train_val, lat_train_val, X_train_val)

if perform_computation:
    f = plt.figure(figsize=(9,4), dpi=120)
    f.add_subplot(1,2,1)
    cvglmnetPlot(lasso_model)
    plt.gca().set_title('Lasso-Regression Model')
    f.add_subplot(1,2,2)
    cvglmnetPlot(ridge_model)
    _ = plt.gca().set_title('Ridge-Regression Model')


if perform_computation:
    lasso_nz_coefs = np.sum(cvglmnetCoef(lasso_model, s = 'lambda_min') != 0)
    ridge_nz_coefs = np.sum(cvglmnetCoef(ridge_model, s = 'lambda_min') != 0)
    print(f'A Total of {lasso_nz_coefs} Lasso-Regression coefficients were non-zero.')
    print(f'A Total of {ridge_nz_coefs} Ridge-Regression coefficients were non-zero.')

def glmnet_elastic(X_train, Y_train, X_test=None, alpha=1):
    """
    Train a elastic-net model using the glmnet library.
    
        Parameters:
                X_train (np.array): A numpy array of the shape (N,d) where N is the number of training data points, and d is the data dimension. 
                Y_train (np.array): A numpy array of the shape (N,) where N is the number of training data points.
                X_test (np.array): A numpy array of the shape (N_test,d) where N_test is the number of testing data points, and d is the data dimension.
                
        Returns:
                fitted_Y_test (np.array): The predicted values on the test data as a numpy array with a shape of (N_test,) where N_test is the number of testing data points.
                glmneet_model (dict): The glmnet library's returned model stored as a python dictionary.
    """        
    if X_test is None:
        X_test = X_train.copy().astype(np.float64)
    # Creating Scratch Variables For glmnet consumption
    X_train = X_train.copy().astype(np.float64)
    Y_train = Y_train.copy().astype(np.float64)
    
    # your code here
#     raise NotImplementedError
    glmnet_model = cvglmnet(x=X_train.copy(), y=Y_train.copy(), family ='gaussian', ptype = 'mse', 
                            nfolds = 10, nlambda=100, alpha=alpha)
    #Note Alpha= 0 it is Lasso, Alpha=1 is Ridge, between 0 and 1 is GLMnet
    
    fitted_Y_test = cvglmnetPredict(glmnet_model, X_test, ptype="response", 
                                    s='lambda_min').reshape((X_test.shape[0],))
    
    assert fitted_Y_test.shape == (X_test.shape[0],), 'fitted_Y should not be two dimensional (hint: reshaping may be helpful)'
    assert isinstance(glmnet_model, dict)
    return fitted_Y_test, glmnet_model

# 2. problem 2
df = pd.read_csv('../GLMnet-lib/credit/credit.csv')
df.head()

X_full = df.iloc[:,:-1].values
Y_full = df.iloc[:,-1].values
X_full.shape, Y_full.shape

outlier_detector = 'LOF'

if outlier_detector == 'LOF':
    outlier_clf = LocalOutlierFactor(novelty=False)
elif outlier_detector == 'IF':
    outlier_clf = IsolationForest(warm_start=True, random_state=12345)
elif outlier_detector == 'EE':
    outlier_clf = EllipticEnvelope(random_state=12345)
else:
    outlier_clf = None

is_not_outlier = outlier_clf.fit_predict(X_full) if outlier_clf is not None else np.ones_like(lat_full)>0
X_useful = X_full[is_not_outlier==1,:]
Y_useful = Y_full[is_not_outlier==1]

X_useful.shape, Y_useful.shape

train_val_indices, test_indices = train_test_split(np.arange(X_useful.shape[0]), test_size=0.2, random_state=12345)

X_train_val = X_useful[train_val_indices, :]
Y_train_val = Y_useful[train_val_indices]

X_test = X_useful[test_indices, :]
Y_test = Y_useful[test_indices]


def glmnet_logistic_elastic(X_train, Y_train, X_test=None, alpha=1):
    """
    Train a elastic-net logistic regression model using the glmnet library.
    
        Parameters:
                X_train (np.array): A numpy array of the shape (N,d) where N is the number of training data points, and d is the data dimension. 
                Y_train (np.array): A numpy array of the shape (N,) where N is the number of training data points.
                X_test (np.array): A numpy array of the shape (N_test,d) where N_test is the number of testing data points, and d is the data dimension.
                alpha (float): The elastic-net regularization parameter
        Returns:
                fitted_Y_test (np.array): The predicted values on the test data as a numpy array with a shape of (N_test,) where N_test is the number of testing 
                                          data points. These values should indicate the prediction classes for test data, and should be either 0 or 1.
                glmneet_model (dict): The glmnet library's returned model stored as a python dictionary.
    """        
    if X_test is None:
        X_test = X_train.copy().astype(np.float64)
    # Creating Scratch Variables For glmnet consumption
    X_train = X_train.copy().astype(np.float64)
    Y_train = Y_train.copy().astype(np.float64)
    
    # your code here
    # Step 1: Train the elastic-net logistic regression model using cross-validation
    glmnet_model = cvglmnet(x=X_train.copy(), y=Y_train.copy(), 
                            family ='binomial', ptype='class', nfolds = 10, nlambda=100, alpha=alpha)
    
    fitted_Y_test  = cvglmnetPredict(glmnet_model, X_test, ptype="class", 
                                     s='lambda_min').reshape((X_test.shape[0],))
    
    
    
    assert fitted_Y_test.shape == (X_test.shape[0],), 'fitted_Y should not be two dimensional (hint: reshaping may be helpful)'
    assert isinstance(glmnet_model, dict)
    return fitted_Y_test, glmnet_model