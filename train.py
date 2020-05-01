import os
import numpy as np 
import tensorflow as tf 

from sklearn.preprocessing import MinMaxScaler
from config import *
from model import *

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

def sliding_window(data,label):
    pass

def main():
    print ('[INFO] Data Input Pipeline')
    txt_list = os.listdir(DATASET_ROOT)
    train_txt = os.path.join(DATASET_ROOT,txt_list[0])
    val_txt = os.path.join(DATASET_ROOT,txt_list[1])
    total_txt = os.path.join(DATASET_ROOT,txt_list[2])
    
    print('Reading: ', total_txt)
    total_dataset = np.loadtxt(total_txt)
    print ('total data shape:', total_dataset.shape)
    train_dataset = total_dataset[:NUM_TRAINING_SAMPLES,:]
    val_dataset = total_dataset[NUM_TRAINING_SAMPLES:,:]
    
    train_data = train_dataset[:,:13]
    train_label = train_dataset[:,-1]
    val_data = val_dataset[:,:13]
    val_label = val_dataset[:,-1]

    # total_data = total_dataset[:,:13]
    # total_label = total_dataset[:,-1]

    # TODO: Sliding window data augmentation
    train_data = np.expand_dims(train_data,axis=2)
    print(train_data.shape,train_label.shape,val_data.shape,val_label.shape)
    
    tf_train = tf.data.Dataset.from_tensor_slices((train_data,train_label))
    tf_val = tf.data.Dataset.from_tensor_slices((val_data,val_label))
    
    

    # tf dataset config
    tf_train = tf_train.shuffle(buffer_size=BUFFER_SIZE)
    tf_train = tf_train.batch(BATCH_SIZE)
    tf_train = tf_train.prefetch(1)

    for data,label in tf_train.take(1):
        print ('Data:',data.shape)

    print('[INFO] Model initialization')
    model = conv1D()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LR),
        loss=tf.losses.CategoricalCrossentropy(), 
        metrics=[[tf.keras.metrics.Accuracy()]]
    )
    print('[INFO] Model training')
    model.fit(
        tf_train,
        epochs = NUM_EPOCHS
    )
if __name__ == '__main__':
    main()