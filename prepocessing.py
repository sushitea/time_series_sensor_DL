import os
import tqdm
import numpy as np 
import tensorflow as tf 

from io import StringIO
from utils import *
from config import *

DATASET_DIR = '/home/xy/research/dataset/OpportunityUCIDataset/dataset'

def read_and_save_data(data,label,target_filename):
    
    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))
    
    try:
        print ('[INFO] reading data file: {0}'.format(data))
        np_data = np.loadtxt(data,delimiter=' ')
        x,y = process_dataset_file(np_data,'gestures') # default to LLA & HL activity
        data_x = np.vstack((data_x, x))
        data_y = np.concatenate([data_y, y])
        obj = (data_x,data_y)
        print ('[INFO] {0}: x.shape:{1}, y.shape:{2}'.format(target_filename,data_x.shape,data_y.shape))
        # with open(os.path.join(DATASET_DIR, target_filename), 'wb') as f:
        #     cp.dump(obj, f)
        # f.close()

    except KeyError as e:
        print(e)

def main():
    data_list = os.listdir(DATASET_DIR)
    data_file_list = []

    for data in data_list:
        data_file_list.append(data)
    
    data_file_list.sort() # sort the list according to alphabetical order
    data_file_list = data_file_list[:-8] # final usage data list

    # create train list and val list
    train_set_list = list(set(data_file_list).intersection(set(TRAIN_LIST)))
    train_set_list.sort()
    val_set_list = list(set(data_file_list).intersection(set(VAL_LIST)))
    val_set_list.sort()

    for i, train_set in enumerate(train_set_list):
        train_set_list[i] = os.path.join(DATASET_DIR,train_set)
    
    for i, val_set in enumerate(val_set_list):
        val_set_list[i] = os.path.join(DATASET_DIR,val_set)

    for train_set in train_set_list:
        read_and_save_data(train_set,LABEL,'LLA_train_no_normalize')
    
    for val_set in val_set_list:
        read_and_save_data(val_set,LABEL,'LLA_test_no_normalize')

if __name__ == "__main__":
    main()