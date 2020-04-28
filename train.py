import os
import numpy as np 
import pandas as pd 

from config import *

DATASET_ROOT = '/home/xy/research/dataset/OpportunityUCIDataset/dataset/RLA_dataset'


def testing():
    root = '/home/xy/research/dataset/OpportunityUCIDataset/dataset/RLA_dataset'
    list_txt = os.listdir(root)
    count = 0

    for txt in list_txt:
        data = np.loadtxt(os.path.join(root,txt))
        count += data.shape[0]
    print(count)

def pd_dataframe_construction(data_list):
    data = np.empty((0,NB_SENSOR_CHANNELS+1))
    for txt in data_list:
        temp_data = np.loadtxt(os.path.join(DATASET_ROOT,txt))
        data = np.concatenate((data,temp_data),axis=0)

    df = pd.DataFrame(data)
    label = df.iloc[:,13]
    print (label.info())

def main():
    print ('[INFO] Data Preprocessing')
    txt_list = os.listdir(DATASET_ROOT)
    train_set_list = list(set(txt_list).intersection(set(TRAIN_TXT_LIST)))
    train_set_list.sort()
    val_set_list = list(set(txt_list).intersection(set(VAL_TXT_LIST)))
    val_set_list.sort()

    pd_dataframe_construction(train_set_list)
if __name__ == '__main__':
    main()