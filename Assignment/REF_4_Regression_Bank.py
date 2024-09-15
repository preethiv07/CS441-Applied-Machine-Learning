%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aml_utils import test_case_checker, perform_computation

# Task 1
def normalize_feats(train_features, some_features):
    """
    Normalizes the sample data features.
    
    Parameters
    ----------
    train_features: A numpy array with the shape (N_train, d), where d is the number of features and N_train is the number of training samples.
    some_features: A numpy array with the shape (N_some, d), where d is the number of features and N_some is the number of samples to be normalized.
    
    Returns
    -------
    some_features_normalized: A numpy array with shape (N_some, d).
    """
    
    # your code here
#     raise NotImplementedError
    mu_train = np.mean(train_features, axis=0)
    sigma_train = np.std(train_features, axis=0)
    some_features_normalized = (some_features - mu_train) / sigma_train
    
    return some_features_normalized

#Task 2

def e_term(x_batch, y_batch, a, b):
    """
    Computes the margin of the data points.
    
    Parameters
    ----------
    x_batch: A numpy array with the shape (N, d), where d is the number of features and N is the batch size.
    y_batch: A numpy array with the shape (N, 1), where N is the batch size.
    a: A numpy array with the shape (d, 1), where d is the number of features. This is the weight vector.
    b: A scalar.
    
    Returns
    -------
    e_batch: A numpy array with shape (N, 1). 
    """
    
    # your code here
#     raise NotImplementedError
    e_batch = 1 - y_batch * (x_batch @ a + b)
    
    return e_batch

# Task 3
def loss_terms_ridge(e_batch, a, lam):
    """
    Computes the hinge and ridge regularization losses.
    
    Parameters
    ----------
    e_batch: A numpy array with the shape (N, 1), where N is the batch size. This is the output of the e_term function you wrote previously, and its kth element is e_k = 1 − y_k(a*x_k+b).
    a: A numpy array with the shape (d, 1), where d is the number of features. This is the weight vector.
    lam: A scalar representing the regularization coefficient 𝜆.
    
    Returns
    -------
    hinge_loss: The hinge regularization loss defined in the above cell.
    ridge_loss: The ridge regularization loss defined in the above cell.
    """
    
    # your code here
    hinge_loss = np.mean(np.maximum(0, e_batch))
    ridge_loss = (lam / 2) * np.sum(a ** 2)
#     ridge_loss = (lam / 2) * (a.T @ a).item()
  
    return np.array((hinge_loss, ridge_loss))


# task 4
def a_gradient_ridge(x_batch, y_batch, e_batch, a, lam):
    """
    Computes the ridge_regularized loss gradient w.r.t the weights vector.
    
    Parameters
    ----------
    x_batch: A numpy array with the shape (N, d), where d is the number of features and N is the batch size.
    y_batch: A numpy array with the shape (N, 1), where N is the batch size.
    e_batch: A numpy array with the shape (N, 1), where N is the batch size. This is the output of the e_term function you wrote previously, and its kth element is e_k = 1 − y_k(a*x_k+b).
    a: A numpy array with the shape (d, 1), where d is the number of features. This is the weight vector.
    lam: A scalar representing the regularization coefficient 𝜆.
    
    Returns
    -------
    grad_a: A numpy array with shape (d, 1) and defined as the gradient of the ridge regularized loss function. 
    """
    
    # your code here
    hinge_grad = -np.mean((e_batch > 0) * (y_batch * x_batch), axis=0).reshape(-1, 1) 
    # My reference: 
    #Hinge loss is calculated only when loss is > than 0.
    #gradient of 𝑦𝑖(𝑎⋅𝑥(𝑖)+𝑏) with respect to "a" is "- (y𝑖 ⋅𝑥(𝑖))"
    #Page 31 in book (>=1 is used or >0)
    
    ridge_grad = lam * a
    grad_a = hinge_grad + ridge_grad
    
    return grad_a

# Task 5
def b_derivative(y_batch, e_batch):
    """
    Computes the loss gradient with respect to the bias parameter b.
    
    Parameters
    ----------
    y_batch: A numpy array with the shape (N, 1), where N is the batch size.
    e_batch: A numpy array with the shape (N, 1), where N is the batch size. This is the output of the e_term function you wrote previously, and its kth element is e_k = 1 − y_k(a*x_k+b).
    
    Returns
    -------
    der_b: A scalar defined as the gradient of the hinge loss w.r.t the bias parameter b.
    """
    
    # your code here
    der_b = -np.mean((e_batch > 0) * y_batch)
    
    return der_b

#Task 6
def loss_terms_lasso(e_batch, a, lam):
    """
    Computes the hinge and lasso regularization losses.
    
    Parameters
    ----------
    e_batch: A numpy array with the shape (N, 1), where N is the batch size. This is the output of the e_term function you wrote previously, and its kth element is e_k = 1 − y_k(a*x_k+b).
    a: A numpy array with the shape (d, 1), where d is the number of features. This is the weight vector.
    lam: A scalar representing the regularization coefficient 𝜆.
    
    Returns
    -------
    hinge_loss: The hinge loss scalar as defined in the cell above.
    lasso_loss: The lasso loss scalar as defined in the cell above.
    """
    
    # your code here
   # your code here
    hinge_loss = np.mean(np.maximum(0, e_batch))
    lasso_loss = (lam) * np.sum(np.abs(a))
    # remember to use abs (if else, throws error)

    
    return np.array((hinge_loss, lasso_loss))

# Task 7
