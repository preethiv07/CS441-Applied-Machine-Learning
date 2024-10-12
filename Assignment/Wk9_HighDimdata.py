%matplotlib inline
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os
from zipfile import ZipFile
import shutil
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

from aml_utils import test_case_checker, perform_computation

# Let's extract the data
with ZipFile('../HiDimClassification-lib/hmpdata.zip', 'r') as zipObj:
    zipObj.extractall()

# Loading the data into lists of lists
col_labels = ['X','Y','Z']
raw_txt_files = []
activity_labels = ['Liedown_bed', 'Walk', 'Eat_soup', 'Getup_bed', 'Descend_stairs', 
                   'Use_telephone', 'Standup_chair', 'Brush_teeth', 'Climb_stairs', 
                   'Sitdown_chair', 'Eat_meat', 'Comb_hair', 'Drink_glass', 'Pour_water']

for activity in activity_labels:
    activity_txts = []
    for file in os.listdir('./HMP_Dataset/'+activity):
        txtdf = pd.read_csv('./HMP_Dataset/'+activity+'/'+file, names=col_labels,  sep=" ")
        activity_txts.append(txtdf)
    raw_txt_files.append(activity_txts)

# Let's clean up after we're done
shutil.rmtree('./HMP_Dataset')

print('Number of samples for each activity:')
for activity, activity_txts in zip(activity_labels, raw_txt_files):
    print(f'    {activity}: {len(activity_txts)}')
print(f'Total number of samples: {sum(len(activity_txts) for activity_txts in raw_txt_files)}')


# Create Test- training set
test_portion = 0.2

np_random = np.random.RandomState(12345)
train_val_txt_files = []
test_txt_files = []
for _,activity_txt_files in enumerate(raw_txt_files):
    num_txt_files = len(activity_txt_files)
    shuffled_indices = np.arange(num_txt_files)
    np_random.shuffle(shuffled_indices)
    
    train_val_txt_files.append([])
    test_txt_files.append([])
    for i, idx in enumerate(shuffled_indices):
        if i < test_portion * num_txt_files:
            test_txt_files[-1].append(activity_txt_files[idx])
        else:
            train_val_txt_files[-1].append(activity_txt_files[idx])

# 1. Training
d = 32
k = 100
train_txt_files = train_val_txt_files

# task 1
def quantize(X, d=32):
    """
    Performs vector quantization.

        Parameters:
                X (np,array): Dimension N x 3
                d (int): The number of samples in the target output 

        Returns:
                out (np.array): A numpy array with the a shape: num columns = 3*d. 
                This array contains the quantized values of the original X matrix.
    """   
    assert X.ndim == 2
    assert X.shape[1] == 3
    
    # your code here
    N = X.shape[0]
    
    outCol = 3*d
    outRow = 3*N // outCol
    
    newX = X[:(outRow*outCol//3), :]
    
    out = newX.reshape(outRow, outCol)
    
    assert out.shape[1] == 3*d
    return out