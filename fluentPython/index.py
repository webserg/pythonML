import numpy as np
def insert_ones(y, segment_end_ms):
    """
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 followinf labels should be ones.


    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms

    Returns:
    y -- updated labels
    """

    # duration of the background (in terms of spectrogram time-steps)
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    # Add 1 to the correct index in the background label (y)
    ### START CODE HERE ### (≈ 3 lines)
    for i in range(0, Ty):
        if segment_end_y > i+1 and i <= segment_end_y+50:
            print(segment_end_y)
            print(i)
            y[0, i] = 1
    ### END CODE HERE ###

    return y


Ty = 1375
arr1 = insert_ones(np.zeros((1, Ty)), 9700)
print(arr1)
print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])