import os
import csv
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from datetime import datetime
from sklearn.metrics import classification_report,confusion_matrix

import torch
from utils import *
from configs import *
from models import DeepConvLSTM


def main():
    # --- initialize training device --- #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = 'fullsensor'
    exp_name = 'noNULL_cross_entropy_stride1_fullsensor'
    remove_null = True
    print('device:', device)
    
    if mode == 'upperbody':
        n_channels = 45
    elif mode == 'leftupperbody' or mode == 'rightupperbody':
        n_channels = 27
    elif mode == 'fullsensor':
        n_channels = 113

    # --- initialize dataset --- #
    X_train, y_train = load_data(mode,'train',len_seq,stride,remove=remove_null)
    X_val, y_val = load_data(mode,'val',len_seq,stride,remove=remove_null)
    # x_train_NULL, y_train_NULL = get_null_data(mode,'train',len_seq,stride, 0)
    # x_val_NULL, y_val_NULL = get_null_data(mode,'val',len_seq,stride, 0)


    train_stats = np.unique([a for y in y_val for a in y],return_counts=True)[1]
    val_stats = np.unique([a for y in y_val for a in y],return_counts=True)[1]
    print('Training set statistics:')
    print(len(train_stats),'classes with distribution',train_stats)
    print('Validation set statistics:')
    print(len(val_stats),'classes with distribution',val_stats)

    # --- initialize model --- #
    net = DeepConvLSTM(n_channels=n_channels, n_classes=17)
    net.apply(init_weights)
    net.to(device)

    train_on_gpu = torch.cuda.is_available()
    
    weight_decay = 1e-5*lr*batch_size*(50/batchlen)
    opt = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay,amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(opt,100) # Learning rate scheduler to reduce LR every 100 epochs

    weights = torch.tensor([max(train_stats)/i for i in train_stats])
    print('weights:',len(weights))
    if train_on_gpu:
        weights = weights.cuda()

    criterion = nn.CrossEntropyLoss(weight=weights)
    criterion.to(device)
    val_criterion = nn.CrossEntropyLoss()
    val_criterion.to(device)
    # criterion = FocalLoss(class_num=17,gamma=4)
    # val_criterion = FocalLoss(class_num=17,gamma=4)
    early_stopping = EarlyStopping(patience=patience, verbose=False)
   
    # --- check if log.csv available, if available,then remove --- #
    csv_path = '/home/xy/research/master-research/sensor-based-human-activity-recognition/log.csv'
    if os.path.isfile(csv_path):
        os.remove(csv_path)

    # --- training --- #
    print('Starting training at',datetime.now())
    start_time=datetime.now()

    with open('log.csv', 'w', newline='') as csvfile:
        for e in range(num_epochs):
            train_losses = []
            net.train() # Setup network for training
            for batch in iterate_minibatches_2D(X_train, y_train, num_batches=num_batches, batchsize=1000, stride=stride, batchlen=batchlen, drop_last=True, shuffle=True):
                x,y,pos= batch
                inputs, targets = torch.from_numpy(x), torch.from_numpy(y) # Get torch tensors.
                opt.zero_grad() # Clear gradients in optimizer
                if pos==0:
                    h = net.init_hidden(inputs.size()[0]) # If we are at the beginning of a metabatch, init lstm hidden states.
                h = tuple([each.data for each in h])  # Get rid of gradients attached to hidden and cell states of the LSTM       
                if (torch.cuda.is_available()):
                    inputs,targets = inputs.cuda(),targets.cuda()
                output, h = net(inputs,h,inputs.size()[0]) # Run inputs through network
                loss = criterion(output.double(), targets.long()) 
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
            val_losses = []
            net.eval() # Setup network for evaluation
            top_classes = []
            targets_cumulative = []
            with torch.no_grad():
                for batch in iterate_minibatches_2D(X_val, y_val, num_batches=num_batches_val, batchsize=val_batch_size, stride=stride, batchlen=batchlen, drop_last=False, shuffle=True):
                    x,y,pos=batch
                    inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
                    targets_cumulative.extend([y for y in y])
                    if pos == 0:
                        val_h = net.init_hidden(inputs.size()[0]) # Init lstm at start of each metabatch
                    if (torch.cuda.is_available()):
                        inputs,targets = inputs.cuda(),targets.cuda()
                    output, val_h = net(inputs,val_h,inputs.size()[0])
                    val_loss = val_criterion(output, targets.long())
                    val_losses.append(val_loss.item())
                    top_p, top_class = output.topk(1,dim=1)
                    top_classes.extend([top_class.item() for top_class in top_class.cpu()])
            equals = [top_classes[i] == target for i,target in enumerate(targets_cumulative)]
            val_accuracy = np.mean(equals)

            f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
            f1macro = metrics.f1_score(targets_cumulative, top_classes, average='macro')
            scheduler.step()
            print('Epoch {}/{}, Train loss: {:.4f}, Val loss: {:.4f}, Acc: {:.2f}, f1: {:.2f}, Macro f1: {:.2f}'.format(e+1,num_epochs,np.mean(train_losses),np.mean(val_losses),val_accuracy,f1score,f1macro))
            writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([np.mean(train_losses),np.mean(val_losses),val_accuracy,f1score,f1macro])
            early_stopping(np.mean(val_losses), net)
            if early_stopping.early_stop:
                print("Stopping training, validation loss has not decreased in {} epochs.".format(patience))
                break

    print('Training finished at ',datetime.now())
    print('Total time elapsed during training:',(datetime.now()-start_time).total_seconds(),'seconds')
    
    X_test, y_test = load_data(mode,'test',len_seq,stride=1,remove=remove_null)

    print('Starting testing at', datetime.now())
    start_time=datetime.now()
    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss(class_num=17,gamma=4)
    if(train_on_gpu):
        net.cuda()
    net.eval()

    val_losses = []
    test_accuracy=0
    f1score=0
    f1macro=0
    targets_cumulative = []
    top_classes = []

    with torch.no_grad():
        for batch in iterate_minibatches_test(X_test, y_test, len_seq, stride=1):
            x,y,pos=batch
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            targets_cumulative.extend([y for y in y])

            if(train_on_gpu):
                targets,inputs = targets.cuda(),inputs.cuda()
            if pos == 0:
                test_h = net.init_hidden(inputs.size()[0])

            output, test_h = net(inputs,test_h,inputs.size()[0])
            val_loss = criterion(output, targets.long())
            val_losses.append(val_loss.item())
            top_p, top_class = output.topk(1,dim=1)
            top_classes.extend([p.item() for p in top_class])

    print('Finished testing at', datetime.now())
    print('Total time elapsed during testing:', (datetime.now()-start_time).total_seconds(),'seconds')

    f1score = metrics.f1_score(targets_cumulative, top_classes, average='weighted')
    classreport = classification_report(targets_cumulative, top_classes,target_names=opp_class_names_mod)
    confmatrix = confusion_matrix(targets_cumulative, top_classes,normalize='true')
    print('#--- TESTING REPORT ---#')
    print(classreport)

    plot_data(exp_name=exp_name, save_fig=True)
    df_cm = pd.DataFrame(confmatrix, index=opp_class_names_mod,columns=opp_class_names_mod)
    plt.figure(10,figsize=(15,12))
    confusion_matrix_fg = sns.heatmap(df_cm,annot=True,fmt='.2f',cmap='Purples')
    confusion_matrix_figure = confusion_matrix_fg.get_figure()    
    confusion_matrix_figure.savefig("/home/xy/research/master-research/sensor-based-human-activity-recognition/exp_results/{}/confusion_matrix.png".format(exp_name), dpi=400)
    
if __name__ == "__main__":
    main()
