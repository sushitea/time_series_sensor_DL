import os
import zipfile
import argparse
import numpy as np
import _pickle as cp
import urllib.request

from io import BytesIO
from pandas import Series

NORM_MAX_THRESHOLDS = [
    3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
    3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
    3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
    3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
    10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
    200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
    10000,  10000,  10000,  10000,  250
    ]

NORM_MIN_THRESHOLDS = [
    -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
    -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
    -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
    -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
    -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
    -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
    -10000, -10000, -10000, -10000, -250
    ] 

IMU_MAX_THRESHOLDS = [
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
    3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500
]

IMU_MIN_THRESHOLDS = [
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
    -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000
] 

def select_columns_opp(data,mode='fullsensor'):
    
    if mode == 'fullsensor':
        # included-excluded
        features_delete = np.arange(46, 50)
        features_delete = np.concatenate([features_delete, np.arange(59, 63)])
        features_delete = np.concatenate([features_delete, np.arange(72, 76)])
        features_delete = np.concatenate([features_delete, np.arange(85, 89)])
        features_delete = np.concatenate([features_delete, np.arange(98, 102)])
        features_delete = np.concatenate([features_delete, np.arange(134, 243)])
        features_delete = np.concatenate([features_delete, np.arange(244, 249)])
        return np.delete(data, features_delete, 1)

    elif mode == 'upperbody':
        column_to_discard = np.arange(1,37)
        column_to_discard = np.concatenate([column_to_discard, np.arange(46, 50)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(59, 63)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(72, 76)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(85, 89)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(98, 243)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(244, 249)])
        return np.delete(data, column_to_discard, 1)

    elif mode == 'leftupperbody':
        column_to_discard = np.arange(1,37)
        column_to_discard = np.concatenate([column_to_discard, np.arange(46, 76)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(85, 89)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(98, 243)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(244, 249)])
        return np.delete(data, column_to_discard, 1)

    elif mode == 'rightupperbody':
        column_to_discard = np.arange(1,37)
        column_to_discard = np.concatenate([column_to_discard, np.arange(46, 50)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(59, 63)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(72, 243)])
        column_to_discard = np.concatenate([column_to_discard, np.arange(244, 249)])
        return np.delete(data, column_to_discard, 1)

def normalize(data, max_list, min_list):
    max_list, min_list = np.array(max_list), np.array(min_list)
    diffs = max_list - min_list
    for i in np.arange(data.shape[1]):
        data[:, i] = (data[:, i]-min_list[i])/diffs[i]
    #     Checking the boundaries
    data[data > 1] = 0.99
    data[data < 0] = 0.00
    return data

def divide_x_y(data, label):
    data_x = data[:, 1:-2]
    if label not in ['locomotion', 'gestures']:
            raise RuntimeError("Invalid label: '%s'" % label)
    if label == 'locomotion':
        data_y = data[:, -2]  # Locomotion label
    elif label == 'gestures':
        data_y = data[:, -1]  # Gestures label

    return data_x, data_y

def adjust_idx_labels(data_y, label):
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

def check_data(data_set):
    print('Checking dataset {0}'.format(data_set))
    data_dir, data_file = os.path.split(data_set)
    # When a directory is not provided, check if dataset is in the data directory
    if data_dir == "" and not os.path.isfile(data_set):
        new_path = os.path.join(os.path.split(__file__)[0], "data", data_set)
        if os.path.isfile(new_path) or data_file == 'OpportunityUCIDataset.zip':
            data_set = new_path

    if (not os.path.isfile(data_set)) and data_file == 'OpportunityUCIDataset.zip':
        print('... dataset path {0} not found'.format(data_set))
        import urllib
        origin = (
            'https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip'
        )
        if not os.path.exists(data_dir):
            print('... creating directory {0}'.format(data_dir))
            os.makedirs(data_dir)
        print('... downloading data from {0}'.format(origin))
        urllib.request.urlretrieve(origin, data_set)

    return data_dir

def process_dataset_file(data, label, mode, upper_threshold, lower_threshold):
    # mode: fullsensor, upperbody, leftupperbody, rightupperbody
    data = select_columns_opp(data, mode=mode)
    data_x, data_y = divide_x_y(data, label)
    data_y = adjust_idx_labels(data_y, label)
    data_y = data_y.astype(int)
    data_x = np.array([Series(i).interpolate() for i in data_x.T]).T
    data_x[np.isnan(data_x)] = 0
    data_x = normalize(data_x, upper_threshold, lower_threshold)

    return data_x, data_y

def split_dataset():

    train_runs = ['S1-Drill','S1-ADL1','S1-ADL2','S1-ADL3','S1-ADL4','S2-Drill','S2-ADL1','S2-ADL2','S3-Drill','S3-ADL1','S3-ADL2','S2-ADL3','S3-ADL3']
    val_runs = ['S1-ADL5']
    test_runs = ['S2-ADL4','S2-ADL5','S3-ADL4','S3-ADL5']

    train_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in train_runs]
    val_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in val_runs]
    test_files = ['OpportunityUCIDataset/dataset/{}.dat'.format(run) for run in test_runs]

    return train_files,val_files,test_files

def generate_data(dataset, folder_name, label, upper_threshold, lower_threshold):
    data_dir = check_data(dataset)

    train_files, test_files, val_files = split_dataset()

    zf = zipfile.ZipFile(dataset)
    mode = folder_name

    try:
        os.mkdir('data')
    except FileExistsError: # Remove data if already there.
        for file in os.scandir('data'):
            if 'data' in file.name:
                os.remove(file.path)

    print('Generating training files')
    for i,filename in enumerate(train_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> train_data_{}'.format(filename,i))
            x, y = process_dataset_file(data, label, mode, upper_threshold, lower_threshold)
            with open('{}/train_data_{}'.format(folder_name,i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    print('Generating validation files')
    for i,filename in enumerate(val_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> val_data_{}'.format(filename,i))
            x, y = process_dataset_file(data, label, mode, upper_threshold, lower_threshold)
            with open('{}/val_data_{}'.format(folder_name,i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

    print('Generating testing files')
    for i,filename in enumerate(test_files):
        try:
            data = np.loadtxt(BytesIO(zf.read(filename)))
            print('... file {} -> test_data_{}'.format(filename,i))
            x, y = process_dataset_file(data, label, mode, upper_threshold, lower_threshold)
            with open('{}/test_data_{}'.format(folder_name,i),'wb') as f:
                cp.dump((x,y),f)
        except KeyError:
            print('ERROR: Did not find {} in zip file'.format(filename))

def main():
    ROOT = os.getcwd()
    PATH = '/home/xy/dataset/opportunity/OpportunityUCIDataset.zip'
    SAVE_FOLDER_NAME = 'rightupperbody'
    label = 'gestures'

    if SAVE_FOLDER_NAME == 'fullsensor':
        NB_SENSOR_CHANNELS = 113
        max_threshold = NORM_MAX_THRESHOLDS
        min_threshold = NORM_MIN_THRESHOLDS
    elif SAVE_FOLDER_NAME == 'upperbody':
        NB_SENSOR_CHANNELS = 45
        max_threshold = IMU_MAX_THRESHOLDS
        min_threshold = IMU_MIN_THRESHOLDS
    elif SAVE_FOLDER_NAME == 'leftupperbody':
        NB_SENSOR_CHANNELS = 27
        max_threshold = IMU_MAX_THRESHOLDS[:27]
        min_threshold = IMU_MIN_THRESHOLDS[:27]
    elif SAVE_FOLDER_NAME == 'rightupperbody':
        NB_SENSOR_CHANNELS = 27
        max_threshold = IMU_MAX_THRESHOLDS[:27]
        min_threshold = IMU_MIN_THRESHOLDS[:27]
    
    generate_data(PATH,SAVE_FOLDER_NAME,label,max_threshold,min_threshold)

if __name__ == "__main__":
    main()
    