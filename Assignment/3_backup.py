    
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
    
    N, height, width = images_.shape
    images_sbb = np.zeros((N, bb_size, bb_size), dtype=images_.dtype)  # Initialize the output array

    
    # your code here
    for i in range(N):
        # Extract the tight bounding box
        tight_box = images_[i, first_ink_rows[i]:last_ink_rows[i]+1, first_ink_cols[i]:last_ink_cols[i]+1]

        # Resize the tight bounding box to the desired bounding box size (bb_size x bb_size)
        resized_box = resize(tight_box, (bb_size, bb_size), anti_aliasing=True, preserve_range=True)
        
        # Ensure that the dtype of the resized image matches the input dtype
        images_sbb[i] = resized_box.astype(images_.dtype)

        
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll get rid of the dummy dimension
        return images_sbb[0]
    else:
        # Otherwise, we'll just work with what's given
        return images_sbb