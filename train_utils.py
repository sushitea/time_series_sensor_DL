import numpy as np 

def stride_sliding_window(np_array,window_length,window_stride):
    x_list = []
    y_list = []
    n_records = np_array.shape[0]
    remainder = (n_records-window_length) % window_stride
    num_windows = 1 + int((n_records-window_length-remainder)/window_stride)
    for i in range (num_windows):
        x_list.append(np_array[i*window_stride:window_length-1+i*window_stride+1])
    
    return np.array(x_list)

def stride_sliding_window_y(np_array,window_length,window_stride):
    y_list =[]
    y_list.append(np_array[window_length-1::window_stride])
    return np.array(y_list)