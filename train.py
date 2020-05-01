import os
import numpy as np 

from sklearn.preprocessing import MinMaxScaler
from config import *

DATASET_ROOT = '/home/xy/research/dataset/OpportunityUCIDataset/dataset/RLA_dataset'
NUM_TRAINING_SAMPLES = 557963

def testing():
    root = '/home/xy/research/dataset/OpportunityUCIDataset/dataset/RLA_dataset'
    list_txt = os.listdir(root)
    count = 0

    for txt in list_txt:
        data = np.loadtxt(os.path.join(root,txt))
        count += data.shape[0]
    print(count)

def normalization(data,min,max):
    normalized_data = (data - min) / (max-min)
    return normalized_data
    

def pd_dataframe_construction(data_txt):
    data = np.loadtxt(data_txt)
    
    df = pd.DataFrame(data)
    label = df.iloc[:,13]
    print (label.info())

def main():
    print ('[INFO] Data Input Pipeline')
    txt_list = os.listdir(DATASET_ROOT)
    train_txt = os.path.join(DATASET_ROOT,txt_list[0])
    val_txt = os.path.join(DATASET_ROOT,txt_list[1])
    total_txt = os.path.join(DATASET_ROOT,txt_list[2])
    
    print('Reading: ', val_txt)
    total_dataset = np.loadtxt(val_txt)
    print ('total data shape:', total_dataset.shape)

    total_data = total_dataset[:,:13]
    total_label = total_dataset[:,-1]
    print('data shape:', total_data.shape)
    print('label shape:', total_label.shape)

    print('[INFO] Model initialization')

    print('[INFO] Model training')

if __name__ == '__main__':
    main()