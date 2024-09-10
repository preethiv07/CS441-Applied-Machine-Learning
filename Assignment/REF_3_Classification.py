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


# task 5 Getting the Last Inky Rows
# # ASK
# Write the function get_last_ink_row_index that finds the last row containing ink pixels and takes following the arguments:
# is_row_inky: A numpy array with the shape (N, height), and the bool dtype. This is the output of the get_is_row_inky function that you implemented before.
# and returns the following:
# last_ink_rows: A numpy array with the shape (N,), and the int64 dtype.
# last_ink_rows[i] is the index of the last row containing any ink pixel in the ith image. The indices should be zero-based.

def get_last_ink_row_index(is_row_inky):
    """
    Finds the last row containing ink pixels.

        Parameters:
                is_row_inky (np.array): A numpy array with the shape (N, height), and the bool dtype. 
                This is the output of the get_is_row_inky function that you implemented before.
                
        Returns:
                last_ink_rows (np.array): A numpy array with the shape (N,), and the int64 dtype. 
    """
    
    # your code here
    # Find the index of the last True value in each row by reversing the array and using np.argmax
    last_ink_rows = is_row_inky.shape[1] - 1 - np.argmax(np.flip(is_row_inky, axis=1), axis=1)
    
    # Handle cases where there are no inky rows by replacing with -1 for those cases
    last_ink_rows[~np.any(is_row_inky, axis=1)] = -1
    
    return last_ink_rows

# Performing sanity checks on your implementation
assert (get_last_ink_row_index(get_is_row_inky(train_images_threshed[:10,:,:])) == 
        np.array([24, 23, 24, 24, 26, 22, 23, 24, 24, 23])).all()

# Checking against the pre-computed test database
test_results = test_case_checker(get_last_ink_row_index, task_id=5)
assert test_results['passed'], test_results['message']


# Task 6 (The Final "Bounding Box" Pre-processor)
# # ASK
# Write the function get_images_bb that applies the "Bounding Box" pre-processing step and takes the following arguments:
# images: A numpy array with the shape (N,height,width), where
# N is the number of samples and could be anything,
# height is each individual image's height in pixels (i.e., number of rows in each image),
# and width is each individual image's width in pixels (i.e., number of columns in each image).
# Do not assume anything about images's dtype or number of samples.
# bb_size: A scalar with the default value of 20, and represents the desired bounding box size.
# and returns the following:
# images_bb: A numpy array with the shape (N,bb_size,bb_size), and the same dtype as images.
# We have provided a template function that uses the previous functions and only requires you to fill in the missing parts. It also handles the input shapes in an agnostic way.
# Important Note: Make sure that you use the np.roll function for this implementation.

def get_images_bb(images, bb_size=20):
    """
    Applies the "Bounding Box" pre-processing step to images.

        Parameters:
                images (np.array): A numpy array with the shape (N,height,width)
                
        Returns:
                images_bb (np.array): A numpy array with the shape (N,bb_size,bb_size), 
                and the same dtype as images. 
    """
    
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll add a dummy dimension to be consistent
        images_ = images.reshape(1,*images.shape)
    else:
        # Otherwise, we'll just work with what's given
        images_ = images
        
    is_row_inky = get_is_row_inky(images_)
    is_col_inky = get_is_col_inky(images_)
    
    first_ink_rows = get_first_ink_row_index(is_row_inky)
    last_ink_rows = get_last_ink_row_index(is_row_inky)
    first_ink_cols = get_first_ink_col_index(is_col_inky)
    last_ink_cols = get_last_ink_col_index(is_col_inky)
    
    # your code here
    N, height, width = images_.shape

    # Compute the middle inky row and column of the raw image
    inky_middle_row = np.floor((first_ink_rows + last_ink_rows + 1) / 2).astype(int)
    inky_middle_col = np.floor((first_ink_cols + last_ink_cols + 1) / 2).astype(int)

    # The middle of the bounding box
    bb_middle = bb_size // 2

    # Calculate the row and column shifts to align the middle inky pixel
    row_shifts = bb_middle - inky_middle_row
    col_shifts = bb_middle - inky_middle_col

    # Roll the images for row and column shifts
#     rolled_images = np.array([np.roll(np.roll(images_[i], row_shifts[i], axis=0), col_shifts[i], axis=1) for i in range(N)])
    
    # Roll the images for row shifts
    rolled_images = np.stack([np.roll(images_[i], row_shifts[i], axis=0) for i in range(N)])

    # Roll the images for column shifts
    rolled_images = np.stack([np.roll(rolled_images[i], col_shifts[i], axis=1) for i in range(N)])

    # Crop the images to the bounding box size (bb_size x bb_size)
    images_bb = rolled_images[:, :bb_size, :bb_size]

    
    #######
    # My Code ends here
    #######
    
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll get rid of the dummy dimension
        return images_bb[0]
    else:
        # Otherwise, we'll just work with what's given
        return images_bb

train_images_bb = get_images_bb(train_images_threshed)
eval_images_bb = get_images_bb(eval_images_threshed)

def get_images_sbb(images, bb_size=20):
    """
    Applies the "Stretched Bounding Box" pre-processing step to images.

        Parameters:
                images (np.array): A numpy array with the shape (N,height,width)
                
        Returns:
                images_sbb (np.array): A numpy array with the shape (N,bb_size,bb_size), 
                and the same dtype and the range of values as images. 
    """
    
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll add a dummy dimension to be consistent
        images_ = images.reshape(1,*images.shape)
    else:
        # Otherwise, we'll just work with what's given
        images_ = images
        
    is_row_inky = get_is_row_inky(images)
    is_col_inky = get_is_col_inky(images)
    
    first_ink_rows = get_first_ink_row_index(is_row_inky)
    last_ink_rows = get_last_ink_row_index(is_row_inky)
    first_ink_cols = get_first_ink_col_index(is_col_inky)
    last_ink_cols = get_last_ink_col_index(is_col_inky)
    
    #### #### #### #### 
#     Code starts here
    #### #### #### #### 
    N, height, width = images.shape
    # images_sbb = np.zeros((N, bb_size, bb_size))

    images_sbb = np.array([resize(images[n, first_ink_rows[n]:last_ink_rows[n]+1, first_ink_cols[n]:last_ink_cols[n]+1], output_shape=(bb_size, bb_size), preserve_range=True, anti_aliasing=True) for n in range(N)]).astype(np.uint8)
#     print(images_sbb.shape)
    # raise NotImplementedError
   
    #### #### #### #### 
#     Code ENDS here
    #### #### #### #### 
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll get rid of the dummy dimension
        return images_sbb[0]
    else:
        # Otherwise, we'll just work with what's given
        return images_sbb