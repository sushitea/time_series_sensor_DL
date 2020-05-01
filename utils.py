import os
import numpy as np
import _pickle as cp

from io import BytesIO
from pandas import Series

def adjust_idx_labels(data_y, label):
    """Transforms original labels into the range [0, nb_labels-1]
    :param data_y: numpy integer array
        Sensor labels
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer array
        Modified sensor labels
    
    Labels:
    
    1   -   Locomotion   -   Stand
    2   -   Locomotion   -   Walk
    4   -   Locomotion   -   Sit
    5   -   Locomotion   -   Lie
    406516   -   ML_Both_Arms   -   Open Door 1
    406517   -   ML_Both_Arms   -   Open Door 2
    404516   -   ML_Both_Arms   -   Close Door 1
    404517   -   ML_Both_Arms   -   Close Door 2
    406520   -   ML_Both_Arms   -   Open Fridge
    404520   -   ML_Both_Arms   -   Close Fridge
    406505   -   ML_Both_Arms   -   Open Dishwasher
    404505   -   ML_Both_Arms   -   Close Dishwasher
    406519   -   ML_Both_Arms   -   Open Drawer 1
    404519   -   ML_Both_Arms   -   Close Drawer 1
    406511   -   ML_Both_Arms   -   Open Drawer 2
    404511   -   ML_Both_Arms   -   Close Drawer 2
    406508   -   ML_Both_Arms   -   Open Drawer 3
    404508   -   ML_Both_Arms   -   Close Drawer 3
    408512   -   ML_Both_Arms   -   Clean Table
    407521   -   ML_Both_Arms   -   Drink from Cup
    405506   -   ML_Both_Arms   -   Toggle Switch
    """

    if label == 'locomotion':  # Labels for locomotion are adjusted
        data_y[data_y == 4] = 3
        data_y[data_y == 5] = 4
    elif label == 'gestures':  # Labels for gestures are adjusted
        data_y[data_y == 406516] = 1
        data_y[data_y == 406517] = 2
        data_y[data_y == 404516] = 3
        data_y[data_y == 404517] = 4
        data_y[data_y == 406520] = 5
        data_y[data_y == 404520] = 6
        data_y[data_y == 406505] = 7
        data_y[data_y == 404505] = 8
        data_y[data_y == 406519] = 9
        data_y[data_y == 404519] = 10
        data_y[data_y == 406511] = 11
        data_y[data_y == 404511] = 12
        data_y[data_y == 406508] = 13
        data_y[data_y == 404508] = 14
        data_y[data_y == 408512] = 15
        data_y[data_y == 407521] = 16
        data_y[data_y == 405506] = 17
    return data_y

def process_dataset_file(data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files
    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """

    # Select correct columns
    # data = select_columns_opp(data)
    data_x, data_y = feature_selection(data)
    # print ('data_x: ', data_x.shape, ' data_y: ',data_y.shape)
    # Colums are segmentd into features and labels
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)

    # Perform linear interpolation
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T

    # Remaining missing data are converted to zero
    data_x[np.isnan(data_x)] = 0

    # All sensor channels are normalized
    # data_x = normalize(data_x, NORM_MAX_THRESHOLDS, NORM_MIN_THRESHOLDS)
    return data_x, data_y

def feature_selection(data, label_x='RLA', label_y='gestures'):
    if label_x == 'LLA':
        data_x = data[:,89:102]
    elif label_x == 'RLA':
        data_x = data[:,63:76]
    elif label_x == 'ALL':
        data_x == data_x[:,:]
    
    if label_y == 'gestures':
        data_y = data[:,-1]
    elif label_y == 'locomotion':
        data_y = data[:,243]
    return data_x, data_y