%matplotlib inline
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt

import numpy as np
from skimage.transform import resize
from sklearn.naive_bayes import GaussianNB, BernoulliNB
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os

from aml_utils import show_test_cases, test_case_checker, perform_computation

## Get Data
if os.path.exists('../ClassifyingImages-lib/mnist.npz'):
    npzfile = np.load('../ClassifyingImages-lib/mnist.npz')
    train_images_raw = npzfile['train_images_raw']
    train_labels = npzfile['train_labels']
    eval_images_raw = npzfile['eval_images_raw']
    eval_labels = npzfile['eval_labels']
else:
    import torchvision
    download_ = not os.path.exists('../ClassifyingImages-lib/mnist.npz')
    data_train = torchvision.datasets.MNIST('mnist', train=True, transform=None, target_transform=None, download=download_)
    data_eval = torchvision.datasets.MNIST('mnist', train=False, transform=None, target_transform=None, download=download_)

    train_images_raw = data_train.data.numpy()
    train_labels = data_train.targets.numpy()
    eval_images_raw = data_eval.data.numpy()
    eval_labels = data_eval.targets.numpy()

    np.savez('../ClassifyingImages-lib/mnist.npz', train_images_raw=train_images_raw, train_labels=train_labels, 
             eval_images_raw=eval_images_raw, eval_labels=eval_labels) 
    

    # Know about the data

    print(train_images_raw.shape) # generrates 60000,28,28
    print(train_images_raw.dtype) # dtype('uint8')
    train_labels.shape, train_labels.dtype # ((60000,), dtype('int64'))
    train_labels[:10] #array([5, 0, 4, 1, 9, 2, 1, 3, 1, 4])
    train_images_raw[0].min(), train_images_raw[0].max() #(0, 255)

    # Display one image in the data
for row_im in train_images_raw[0]:
    print(row_im.tolist())
plt.imshow(train_images_raw[0], cmap='Greys')

# Thresholding
# Task 1
# Write the function get_thresholded that does image thresholding and takes following the arguments:
# images_raw: A numpy array. Do not assume anything about its shape, dtype or range of values. Your function should be careless about these attributes.
# threshold: A scalar value.
# and returns the following:
# threshed_image: A numpy array with the same shape as images_raw, and the bool dtype. This array should indicate whether each elemelent of images_raw is greater than or equal to  threshold.

def get_thresholded(images_raw, threshold):
    """
    Perform image thresholding.

        Parameters:
                images_raw (np,array): Do not assume anything about its shape, dtype or range of values. 
                Your function should be careless about these attributes.
                threshold (int): A scalar value.

        Returns:
                threshed_image (np.array): A numpy array with the same shape as images_raw, and the bool dtype. 
                This array should indicate whether each elemelent of images_raw is greater than or equal to 
                threshold.
    """
    
    # your code here
    threshed_image = images_raw >= threshold
    
    return threshed_image 

train_images_threshed = get_thresholded(train_images_raw, threshold=20)
eval_images_threshed = get_thresholded(eval_images_raw, threshold=20)

# 0.3Creating "Bounding Box" Images0.3 

# Task 2 - Finding Inky Rows

# Write the function get_is_row_inky that finds the rows with ink pixels and takes following the arguments:
# images: A numpy array with the shape (N,height,width), where
# N is the number of samples and could be anything,
# height is each individual image's height in pixels (i.e., number of rows in each image),
# and width is each individual image's width in pixels (i.e., number of columns in each image).
# Do not assume anything about images's dtype or the number of samples or the height or the width.
# and returns the following:
# is_row_inky: A numpy array with the shape (N, height), and the bool dtype.
# is_row_inky[i,j] should be True if any of the pixels in the jth row of the ith image was an ink pixel, and False otherwise.

def get_is_row_inky(images):
    """
    Finds the rows with ink pixels.

        Parameters:
                images (np,array): A numpy array with the shape (N, height, width)

        Returns:
                is_row_inky (np.array): A numpy array with the shape (N, height), and the bool dtype. 
    """
    
    # your code here
    is_row_inky = np.any(images > 0, axis=2)
    
    return is_row_inky


# # Task 3 Finding Inky Columns
# Similar to get_is_row_inky, Write the function get_is_col_inky that finds the columns with ink pixels and takes following the arguments:
# images: A numpy array with the shape (N,height,width), where
# N is the number of samples and could be anything,
# height is each individual image's height in pixels (i.e., number of rows in each image),
# and width is each individual image's width in pixels (i.e., number of columns in each image).
# Note: Do not assume anything about images's dtype or the number of samples or the height or the width.
# and returns the following:
# is_col_inky: A numpy array with the shape (N, width), and the bool dtype.
# is_col_inky[i,j] should be True if any of the pixels in the jth column of the ith image was an ink pixel, and False otherwise.
def get_is_col_inky(images):
    """
    Finds the columns with ink pixels.

        Parameters:
                images (np.array): A numpy array with the shape (N,height,width).
                
        Returns:
                is_col_inky (np.array): A numpy array with the shape (N, width), and the bool dtype. 
    """
    
    # your code here
    is_col_inky = np.any(images > 0, axis=1)
    
    return is_col_inky

# Task 4 ( Getting the First Inky Rows
# Write the function get_first_ink_row_index that finds the first row containing ink pixels and takes following the arguments:
# is_row_inky: A numpy array with the shape (N, height), and the bool dtype. This is the output of the get_is_row_inky function that you implemented before.
# and returns the following:
# first_ink_rows: A numpy array with the shape (N,), and the int64 dtype.
# first_ink_rows[i] is the index of the first row containing any ink pixel in the ith image. The indices should be zero-based.)

def get_first_ink_row_index(is_row_inky):
    """
     Finds the first row containing ink pixels

        Parameters:
                is_row_inky (np.array): A numpy array with the shape (N, height), and the bool dtype. 
                This is the output of the get_is_row_inky function that you implemented before.
                
        Returns:
                first_ink_rows (np.array): A numpy array with the shape (N,), and the int64 dtype. 
    """
    
    # your code here
#     raise NotImplementedError
# Find the index of the first True value in each row (axis=1), if no True exists return -1
    first_ink_rows = np.argmax(is_row_inky, axis=1)
    
    # Handle cases where there are no inky rows by replacing 0 with -1 for those cases
    first_ink_rows[~np.any(is_row_inky, axis=1)] = -1
    
    return first_ink_rows

def get_first_ink_col_index(is_col_inky):
    return get_first_ink_row_index(is_col_inky)
