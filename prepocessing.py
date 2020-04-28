import os
import tqdm
import numpy as np 
import tensorflow as tf 

from io import StringIO
from utils import *
from config import *

DATASET_DIR = '/home/xy/research/dataset/OpportunityUCIDataset/dataset'
SAVE_TXT_DIR = '/home/xy/research/dataset/OpportunityUCIDataset/dataset/RLA_dataset'
TRAIN_TARGET_FILENAME = 'RLA_train.txt'
VAL_TARGET_FILENAME = 'RLA_val.txt'
LABEL = 'gestures'

def read_data(data,label):
    
    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))
    
    try:
        print ('[INFO] reading data file: {0}'.format(data))
        filename = data.split('/')[-1].split('.')[0]
        np_data = np.loadtxt(data, delimiter=' ')
        x,y = process_dataset_file(np_data,label) # desfault to LLA & HL activity
        data_x = np.vstack((data_x, x))
        data_y = np.concatenate([data_y, y])
        data_y = np.expand_dims(data_y,axis =1)
        final_data = np.concatenate((data_x,data_y),axis=1)
        
    except KeyError as e:
        print(e)

    print ('[INFO] {0}: x.shape:{1}, y.shape:{2}'.format(filename,data_x.shape,data_y.shape))

    return final_data

def save_np_data(data,target_filename):
    dir_check = os.path.isdir(SAVE_TXT_DIR)
    if dir_check == False:
        os.mkdir(SAVE_TXT_DIR)
  
    np.savetxt(os.path.join(SAVE_TXT_DIR,target_filename),data,fmt='%1.4f')
    print('[INFO] {0} is saved.'.format(target_filename))

def main():
    data_list = os.listdir(DATASET_DIR)
    data_file_list = []
    train_data = np.empty((0, NB_SENSOR_CHANNELS+1))
    val_data = np.empty((0, NB_SENSOR_CHANNELS+1))

    for data in data_list:
        data_file_list.append(data)
    
    data_file_list.sort() # sort the list according to alphabetical order
    data_file_list = data_file_list[:-8] # final usage data list (left out with s4 data)

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
        temp_train_data = read_data(train_set,LABEL)
        train_data = np.concatenate((train_data,temp_train_data),axis=0)

    for val_set in val_set_list:
        temp_val_data = read_data(val_set,LABEL)
        val_data = np.concatenate((val_data,temp_val_data),axis=0)

    print('Train size: ',train_data.shape)
    print('Val size: ',val_data.shape)
    save_np_data(train_data,TRAIN_TARGET_FILENAME)
    save_np_data(val_data,VAL_TARGET_FILENAME)
    print('[INFO] Preprocessing done. Continue to train.')

if __name__ == "__main__":
    main()