# Attention:
# This assignment is computationally heavy, and inefficient implementations may not pass the autograding even if they technically produce the correct results. To avoid this, make sure you read and understand all the instructions before starting to implement the tasks. Failure to follow the instructions closely will most likely cause timeouts.
# It is your responsibility to make sure your implementation is not only correct, but also as efficient as possible. If you follow all the instructions provided, you should be able to have all the cells evaluated in under 10 minutes.

# Summary
# CIFAR-10 is a dataset of 32x32 images in 10 categories,
# collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. It is often used to evaluate machine learning algorithms. 
# You can download this dataset from https://www.cs.toronto.edu/~kriz/cifar.html.
# For each category, compute the mean image and the first 20 principal components. 
# Plot the error resulting from representing the images of each category using the first 20 principal components against 
# the category.
# Compute the distances between mean images for each pair of classes. Use principal coordinate analysis to make a 2D map 
# of the means of each categories. For this exercise, compute distances by thinking of the images as vectors.
# Here is another measure of the similarity of two classes. For class A and class B, define E(A | B) to be the 
# average error obtained by representing all the images of class A using the mean of class A 
# and the first 20 principal components of class B. Now define the similarity between classes to be 
# (1/2)(E(A | B) + E(B | A)). If A and B are very similar, then this error should be small, because A's 
# principal components should be good at representing B. But if they are very different, then A's principal components
# should represent B poorly. In turn, the similarity measure should be big. Use principal coordinate analysis to make a 
# 2D map of the classes. Compare this map to the map in the previous exercise? are they different? why?

if os.path.exists('../PCA-lib/cifar10.npz'):    
    np_file = np.load('../PCA-lib/cifar10.npz')
    train_images_raw = np_file['train_images_raw']
    train_labels = np_file['train_labels']
    eval_images_raw = np_file['eval_images_raw']
    eval_labels = np_file['eval_labels']
else:
    import torchvision
    import shutil
    download_ = not os.path.exists('../PCA-lib/cifar10/')
    data_train = torchvision.datasets.CIFAR10('../PCA-lib/cifar10', train=True, transform=None, target_transform=None, download=download_)
    data_eval = torchvision.datasets.CIFAR10('../PCA-lib/cifar10', train=False, transform=None, target_transform=None, download=download_)
    shutil.rmtree('../PCA-lib/cifar10/')
    train_images_raw = data_train.data
    train_labels = np.array(data_train.targets)
    eval_images_raw = data_eval.data
    eval_labels = np.array(data_eval.targets)
    np.savez('../PCA-lib/cifar10.npz', train_images_raw=train_images_raw, train_labels=train_labels, 
             eval_images_raw=eval_images_raw, eval_labels=eval_labels)
    

    class_to_idx = {'airplane': 0,
                'automobile': 1,
                'bird': 2,
                'cat': 3,
                'deer': 4,
                'dog': 5,
                'frog': 6,
                'horse': 7,
                'ship': 8,
                'truck': 9}
    
images_raw = np.concatenate([train_images_raw, eval_images_raw], axis=0)
labels = np.concatenate([train_labels, eval_labels], axis=0)
images_raw.shape, labels.shape


def pca_mse(data_raw, num_components=20):
    
    # your code here
    # Step 1: Reshape the data into shape (N, d)
    N = data_raw.shape[0]  # Number of samples
    d = np.prod(data_raw.shape[1:])  # Flatten remaining dimensions (e.g., 32x32x3)
    X = data_raw.reshape(N, d)
    
    # Step 2: Center the data by subtracting the mean of each feature (to get zero mean)
    X = X - np.mean(X, axis=0)
    
    # Step 3: Perform SVD on the centered data (Only compute singular values)
    S_x = np.linalg.svd(X, compute_uv=False)
    
    # Step 4: Calculate the variance explained by each component (squared singular values)
    singular_values_squared = S_x ** 2
    
    # Step 5: Calculate the total variance by summing all the squared singular values
    total_variance = np.sum(singular_values_squared)
    
    # Step 6: Calculate the variance explained by the top 'num_components' components
    variance_retained = np.sum(singular_values_squared[:num_components])
    
    # Step 7: Calculate the mean squared error from the components that were dropped
    variance_dropped = total_variance - variance_retained
    mse = variance_dropped / N  # Divide by the number of samples to get MSE

    
    assert np.isscalar(mse)
    return np.float64(mse)

if perform_computation:
    class_names = []
    class_mses = []
    for cls_name, cls_label in class_to_idx.items():
        data_raw = images_raw[labels == cls_label,:,:,:]
        start_time = time.time()
        print(f'Processing Class {cls_name}', end='')
        cls_mse = pca_mse(data_raw, num_components=20)
        print(f' (The SVD operation took %.3f seconds)' % (time.time()-start_time))
        class_names.append(cls_name)
        class_mses.append(cls_mse)

if perform_computation:
    fig, ax = plt.subplots(figsize=(9,4.), dpi=120)
    sns.barplot(class_names, class_mses, ax=ax)
    ax.set_title('The Mean Squared Error of Representing Each Class by the Principal Components')
    ax.set_xlabel('Class')
    _ = ax.set_ylabel('Mean Squared Error')

# @ 2. principal Co-ordinate analysis
class_mean_list = []
for cls_label in sorted(class_to_idx.values()):
    data_raw = images_raw[labels == cls_label,:,:,:]
    class_mean = np.mean(data_raw, axis=0).reshape(1,-1)
    class_mean_list.append(class_mean)
class_means = np.concatenate(class_mean_list, axis=0)