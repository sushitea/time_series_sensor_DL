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
    print ('[INFO] Data Preprocessing')
    txt_list = os.listdir(DATASET_ROOT)
    train_txt = os.path.join(DATASET_ROOT,txt_list[0])
    val_txt = os.path.join(DATASET_ROOT,txt_list[1])

    train_data = np.loadtxt(train_txt)
    val_data = np.loadtxt(val_txt)
    print ('train data:', train_data.shape)
    print ('val data:', val_data.shape)
    total_dataset = np.concatenate((train_data,val_data),axis=0)
    total_data = total_dataset[:,:13]
    total_label = total_dataset[:,-1:]

    scaler = MinMaxScaler()

    min_col = total_data.min(axis=0)
    max_col = total_data.max(axis=0)

    # print(total_data.shape)
    # print(total_label, total_label.shape)

    sklearn_normalized_data = scaler.transform(total_data)
    print('sklearn_normalized_data:' ,sklearn_normalized_data)
    print('shape:', sklearn_normalized_data.shape)
    
    # save sklearn normalized data
    # np.savetxt('sklearn_normalized_data.txt',sklearn_normalized_data,fmt='%1.4f')
    # print(mean_data.shape)






    # pd_dataframe_construction(train_set_list)


if __name__ == '__main__':
    main()