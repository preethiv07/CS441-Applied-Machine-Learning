# WORKING!
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
    inky_middle_row = np.floor(((first_ink_rows + last_ink_rows + 1) / 2))
    inky_middle_col = np.floor(((first_ink_cols + last_ink_cols + 1) / 2))
    bb_middle = np.floor(bb_size / 2)
    
    
    images_bb = np.zeros(shape=(images_.shape[0], bb_size, bb_size), dtype=np.uint8)
    
    for i in range(images_.shape[0]):
        # First choose single image from the list
        temp_image = images_[i]
        
        # Then roll the images
        temp1 = np.roll(temp_image, (bb_middle-inky_middle_row[i]).astype(int), axis=0)
        temp2 = np.roll(temp1, (bb_middle-inky_middle_col[i]).astype(int), axis=1)
        
        # 'reshape' it to the desired size
        temp_image = temp2[:bb_size, :bb_size]
        
        images_bb[i] = temp_image


    
    #######
    # My Code ends here
    #######
    
    if len(images.shape)==2:
        # In case a 2d image was given as input, we'll get rid of the dummy dimension
        return images_bb[0]
    else:
        # Otherwise, we'll just work with what's given
        return images_bb