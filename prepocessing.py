import os
import tqdm
import numpy as np 
import tensorflow as tf 

from sklearn.preprocessing import MinMaxScaler
from io import StringIO
from utils import *

DATASET_DIR = '/home/xy/research/dataset/OpportunityUCIDataset/dataset'
SAVE_TXT_DIR = '/home/xy/research/dataset/OpportunityUCIDataset/dataset/RLA_dataset'
TRAIN_TARGET_FILENAME = 'RLA_train.txt'
VAL_TARGET_FILENAME = 'RLA_val.txt'
TOTAL_TARGET_FILENAME = 'RLA_total.txt'
LABEL = 'gestures'
NB_SENSOR_CHANNELS = 13

TRAIN_LIST = [
    'S1-Drill.dat',
    'S1-ADL1.dat',
    'S1-ADL2.dat',
    'S1-ADL3.dat',
    'S1-ADL4.dat',
    'S1-ADL5.dat',
    'S2-Drill.dat',
    'S2-ADL1.dat',
    'S2-ADL2.dat',
    'S2-ADL3.dat',
    'S3-Drill.dat',
    'S3-ADL1.dat',
    'S3-ADL2.dat',
    'S3-ADL3.dat'
]

VAL_LIST = [
    'S2-ADL4.dat',
    'S2-ADL5.dat',
    'S3-ADL4.dat',
    'S3-ADL5.dat'
]

def read_data(data,label):
    
    data_x = np.empty((0, NB_SENSOR_CHANNELS))
    data_y = np.empty((0))
    
    try:
        print ('[INFO] reading data file: {0}'.format(data))
        filename = data.split('/')[-1].split('.')[0]
        np_data = np.loadtxt(data, delimiter=' ')
        x,y = process_dataset_file(np_data,label) # desfault to LLA & ML_both hand
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
    print('[INFO] Normalize Dataset.')

    # Total data construction
    total_dataset = np.concatenate((train_data,val_data), axis=0)
    total_data = total_dataset[:,:13]
    total_label = total_dataset[:,13]
    total_label = np.expand_dims(total_label, axis=1)
    # print('Total data:', total_data.shape, ', Total label:', total_label.shape)
    # print('Total dataset size:', total_dataset.shape)

    # Total data normalization
    # TODO: check if the normalization is based on full array? Normalization should be done using each 3 columns
    scaler = MinMaxScaler()
    scaler.fit(total_data)
    normalized_total_data = scaler.transform(total_data)
    final_data = np.concatenate((normalized_total_data,total_label),axis=1)
    print ('Final Data shape:', final_data.shape)
    save_np_data(val_data,TOTAL_TARGET_FILENAME)

if __name__ == "__main__":
    main()