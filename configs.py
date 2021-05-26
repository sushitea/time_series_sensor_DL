opp_class_names = ['Null','Open Door 1','Open Door 2','Close Door 1','Close Door 2','Open Fridge',
'Close Fridge','Open Dishwasher','Close Dishwasher','Open Drawer 1','Close Drawer 1','Open Drawer 2','Close Drawer 2',
'Open Drawer 3','Close Drawer 3','Clean Table','Drink from Cup','Toggle Switch']

opp_class_names_mod = ['Open Door 1','Open Door 2','Close Door 1','Close Door 2','Open Fridge',
'Close Fridge','Open Dishwasher','Close Dishwasher','Open Drawer 1','Close Drawer 1','Open Drawer 2','Close Drawer 2',
'Open Drawer 3','Close Drawer 3','Clean Table','Drink from Cup','Toggle Switch']

len_seq = 24 
stride = 1 
num_epochs = 100 
num_batches= 20
batch_size = 1000 
patience= 10
batchlen = 50
val_batch_size = 1000 
test_batch_size = 10000 
lr = 0.0001 
num_batches_val = 1 
lr_step = 100

# config for data_processing
