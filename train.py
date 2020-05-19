import os
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf 

from sklearn.preprocessing import MinMaxScaler
from model import *
from train_utils import *
from train_config import *

DATASET_ROOT = '/home/xy/research/dataset/OpportunityUCIDataset/dataset/RLA_dataset'
NUM_TRAINING_SAMPLES = 557963

def main():
    #-----Input Preprocessing-----#
    print ('[INFO] Data Input Pipeline')
    txt_list = os.listdir(DATASET_ROOT)
    train_txt = os.path.join(DATASET_ROOT,txt_list[0])
    val_txt = os.path.join(DATASET_ROOT,txt_list[1])
    total_txt = os.path.join(DATASET_ROOT,txt_list[2])
    
    print('[INFO] Reading: ', total_txt)
    total_dataset = np.loadtxt(total_txt)
    print ('[INFO] Total Data Shape:', total_dataset.shape)
    train_dataset = total_dataset[:NUM_TRAINING_SAMPLES,:]
    val_dataset = total_dataset[NUM_TRAINING_SAMPLES:,:]
    
    train_data = train_dataset[:,:13]
    train_label = train_dataset[:,-1]
    val_data = val_dataset[:,:13]
    val_label = val_dataset[:,-1]

    # TODO: Sliding window data augmentation
    # train_data = np.expand_dims(train_data,axis=2)
    # val_data = np.expand_dims(val_data,axis=2)
    sliding_train_data = stride_sliding_window(train_data,WINDOW_LENGTH,WINDOW_STRIDE)
    sliding_val_data = stride_sliding_window(val_data,WINDOW_LENGTH,WINDOW_STRIDE)
    sliding_train_label = np.squeeze(stride_sliding_window_y(train_label,WINDOW_LENGTH,WINDOW_STRIDE),axis=0)
    sliding_val_label = np.squeeze(stride_sliding_window_y(val_label,WINDOW_LENGTH,WINDOW_STRIDE),axis=0)
    
    print ('[INFO]traindata:{},trainlabel:{},valdata:{},vallabel:{}'
        .format(sliding_train_data.shape,sliding_train_label.shape,sliding_val_data.shape,sliding_val_label.shape))
    
    tf_train = tf.data.Dataset.from_tensor_slices((sliding_train_data,sliding_train_label))
    tf_val = tf.data.Dataset.from_tensor_slices((sliding_val_data,sliding_val_label))

    '''
    # tf dataset config
    tf_train = tf_train.shuffle(buffer_size=BUFFER_SIZE)
    tf_train = tf_train.batch(BATCH_SIZE)
    tf_train = tf_train.prefetch(1)
    tf_val = tf_val.batch(BATCH_SIZE)

    for data,label in tf_train.take(1):
        print ('Data:',data.shape)

    #-----Model Initialization & Config-----#
    print('[INFO] Model initialization')
    model = conv1D()
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LR),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    model.build(INPUT_SHAPE)
    model.summary()

    #-----TF Callbacks-----#
    EXP_NAME = 'No_sliding_window_CNN1D'
    LOG_NAME = EXP_NAME + '_' + str(BATCH_SIZE) + '_' + str(LR) + '_' + str(NUM_EPOCHS)
    log_dir = 'logs/' + LOG_NAME
    tf_callback = [
        # tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='weights.{epoch:02d}-{val_loss:.2f}.hdf5',
            monitor='val_loss',
            verbose=1,
            save_weights_only=False,
            save_best_only=True
        )
    ]

    #-----Model Training-----#
    print('[INFO] Model training')
    print('[INFO] Experiment parameters: ', log_dir)
    history = model.fit(
        tf_train,
        epochs = NUM_EPOCHS,
        validation_data = tf_val,
        verbose=1,
        callbacks=tf_callback
    )

    print('[INFO] Matplotlib Loss Visualization')
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    '''
if __name__ == '__main__':
    main()