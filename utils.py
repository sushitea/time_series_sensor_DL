import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import csv
import time
import itertools
import _pickle as cp
import os
import glob

from numpy.lib.stride_tricks import as_strided as ast
from torch import nn
from torch.autograd import Variable

def init_weights(m):
	if type(m) == nn.LSTM:
		for name, param in m.named_parameters():
			if 'weight_ih' in name:
				torch.nn.init.orthogonal_(param.data)
			elif 'weight_hh' in name:
				torch.nn.init.orthogonal_(param.data)
			elif 'bias' in name:
				param.data.fill_(0)
	elif type(m) == nn.Conv1d or type(m) == nn.Linear:
		torch.nn.init.orthogonal_(m.weight)
		m.bias.data.fill_(0)


def iterate_minibatches(inputs, targets, batchsize, shuffle=True, num_batches=-1):
	batch = lambda j : [x for x in range(j*batchsize,(j+1)*batchsize)]
	batches = [i for i in range(int(len(inputs)/batchsize)-1)]

	if shuffle:
		np.random.shuffle(batches)
		for i in batches[0:num_batches]:
			yield np.array([inputs[i] for i in batch(i)]), np.array([targets[i] for i in batch(i)])
	else:
		for i in batches[0:num_batches]:
			yield np.array([inputs[i] for i in batch(i)]),np.array( [targets[i] for i in batch(i)])


def plot_data(logname='log.csv',exp_name='baseline',save_fig=False):
	train_loss_plot = []
	val_loss_plot = []
	acc_plot = []
	f1_plot = []
	f1_macro = []

	with open(logname, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=' ', quotechar= '|')
		for row in reader:
			train_loss_plot.append(float(row[0]))
			val_loss_plot.append(float(row[1]))
			acc_plot.append(float(row[2]))
			f1_plot.append(float(row[3]))
			f1_macro.append(float(row[4]))

	if save_fig:
		try:
			os.makedirs('exp_results/{}'.format(exp_name))
		except FileExistsError:
			pass

	plt.figure(1)
	plt.title('Training & Validation Loss')
	plt.ylabel('Categorical Cross Entropy Loss')
	plt.xlabel('Epoch')
	plt.plot(train_loss_plot)
	plt.plot(val_loss_plot)
	plt.legend(['Train Loss','Test Loss'], loc='upper right')
	if save_fig:
		plt.savefig('exp_results/{}/TrainVal_loss_{}.png'.format(exp_name,time.time()))

	plt.figure(2)
	plt.title('Validation Accuracy')
	plt.ylabel('Weighted accuracy')
	plt.xlabel('Epoch')
	plt.plot(acc_plot)
	if save_fig:
		plt.savefig('exp_results/{}/Val_acc_{}.png'.format(exp_name,time.time()))

	plt.figure(3)
	plt.title('Validation f1 score')
	plt.ylabel('f1 score')
	plt.xlabel('Epoch')
	plt.plot(f1_plot,label='Weighted')
	plt.plot(f1_macro,label='Macro')
	plt.legend()
	if save_fig:
		plt.savefig('exp_results/{}/Val_f1_{}.png'.format(exp_name,time.time()))

	if not save_fig:
		plt.show()

def discard_null_label(data_y):
	data_y[data_y == 1] = 0
	data_y[data_y == 2] = 1
	data_y[data_y == 3] = 2
	data_y[data_y == 4] = 3
	data_y[data_y == 5] = 4
	data_y[data_y == 6] = 5
	data_y[data_y == 7] = 6
	data_y[data_y == 8] = 7
	data_y[data_y == 9] = 8
	data_y[data_y == 10] = 9
	data_y[data_y == 11] = 10
	data_y[data_y == 12] = 11
	data_y[data_y == 13] = 12
	data_y[data_y == 14] = 13
	data_y[data_y == 15] = 14
	data_y[data_y == 16] = 15
	data_y[data_y == 17] = 16

	return data_y

def binary_classification_label(data_y):
	data_y[data_y == 0] = 0
	data_y[data_y == 1] = 1
	data_y[data_y == 2] = 1
	data_y[data_y == 3] = 1
	data_y[data_y == 4] = 1
	data_y[data_y == 5] = 1
	data_y[data_y == 6] = 1
	data_y[data_y == 7] = 1
	data_y[data_y == 8] = 1
	data_y[data_y == 9] = 1
	data_y[data_y == 10] = 1
	data_y[data_y == 11] = 1
	data_y[data_y == 12] = 1
	data_y[data_y == 13] = 1
	data_y[data_y == 14] = 1
	data_y[data_y == 15] = 1
	data_y[data_y == 16] = 1
	data_y[data_y == 17] = 1

	return data_y

def combine_multiple_label(data_y):
	pass

def remove_data(x,y,label):
    y_pos = np.where(y==label)
    y_pos = np.array(y_pos)
    y_pos = np.squeeze(y_pos, axis=0)
    x_data = np.delete(x, y_pos, 0)
    y_data = np.delete(y, y_pos, 0)
    return x_data, y_data

def get_null_data(mode,name, len_seq, stride, label):
	Xs = []
	ys = []

	## Use glob module and wildcard to build a list of files to load from data directory
	path = "{}/{}_data_*".format(mode,name)
	data = glob.glob(path)

	for file in data:
		X, y = load_dataset(file)
		y_pos = np.where(y==label)
		y_pos = np.array(y_pos)
		y_pos = np.squeeze(y_pos, axis=0)
		ratioNum = round(len(y_pos) * 0.1)
		index = np.random.choice(y_pos, ratioNum, replace=False)
		x_random = X[index]
		y_random = y[index]
		X, y = slide(x_random, y_random, len_seq, stride, save=False)
		Xs.append(X)
		ys.append(y)

	return Xs, ys

def load_data(mode,name,len_seq,stride,remove=False):
	Xs = []
	ys = []

	## Use glob module and wildcard to build a list of files to load from data directory
	path = "{}/{}_data_*".format(mode,name)
	data = glob.glob(path)

	for file in data:
		X, y = load_dataset(file)
		if remove == True:
			X, y = remove_data(X, y, 0)
			y = discard_null_label(y)
		X, y = slide(X, y, len_seq, stride, save=False)
		Xs.append(X)
		ys.append(y)

	return Xs, ys

def load_data_resampling(mode,name,len_seq,stride,remove=False):
	Xs = []
	ys = []

	## Use glob module and wildcard to build a list of files to load from data directory
	path = "{}/{}_data_*".format(mode,name)
	data = glob.glob(path)

	for file in data:
		X, y = load_dataset(file)
		if remove == True:
			X, y = remove_data(X, y, 0)
			# y = discard_null_label(y)
		X, y = slide(X, y, len_seq, stride, save=False)
		Xs.append(X)
		ys.append(y)

	return Xs, ys

def load_data_mod(mode,name,len_seq,stride,remove=False):
	Xs = []
	ys = []

	## Use glob module and wildcard to build a list of files to load from data directory
	path = "{}/{}_data_*".format(mode,name)
	data = glob.glob(path)

	for file in data:
		X, y = load_dataset(file)
		y = binary_classification_label(y)
		X, y = slide(X, y, len_seq, stride, save=False)
		Xs.append(X)
		ys.append(y)

	return Xs, ys

def load_dataset(filename):

	with open(filename, 'rb') as f:
		data = cp.load(f)

	X, y = data

	print('Got {} samples from {}'.format(X.shape, filename))

	X = X.astype(np.float32)
	y = y.astype(np.uint8)


	return X, y

def slide(data_x, data_y, ws, ss,save=False):
	x = sliding_window(data_x, (ws,data_x.shape[1]),(ss,1))
	y = np.asarray([i[-1] for i in sliding_window(data_y, ws, ss)]).astype(np.uint8)

	if save:
		with open('data_slid','wb') as f:
			cp.dump((x,y),f,protocol=4)

	else:
		return x,y


def iterate_minibatches_2D(inputs, targets, batchsize=1000, stride=1, num_batches=20, batchlen=50, drop_last=True, shuffle=True):
	"""
	Args:
		inputs (array): Dataset sensor channels, a stacked array of runs, after a sliding window has been applied.
		targets (array): Dataset labels, a stacked array of labels corresponding to the windows in inputs.
		batchsize (int): Number of windows in each batch.
		stride (int): Size of sliding window step.
		num_batches (int): Number of metabatches to return before finishing the epoch.
				Default: 10
		batchlen (int): Number of contiguous windows per batch. 
				Default: 50
		drop_last (bool): Whether to drop the last incomplete batch when dataset does not divide neatly by batchsize.
				Default: True
		shuffle (bool): Determines whether to shuffle the batches or iterate through sequentially.
				Default: True
	"""
	
	window_size = len(inputs[0][0])
	assert (window_size/stride).is_integer(), 'in order to generate sequential batches, the sliding window length must be divisible by the step.'
	starts = [[x for x in range(0,len(i)-int(((batchlen*window_size)+1)/stride))] for i in inputs]
	for i in range(1,len(starts)):
		starts[i] = [x+1+starts[i-1][-1]+int(((batchlen*window_size)+1)/stride) for x in starts[i]]
	starts = [val for sublist in starts for val in sublist]
	inputs = [val for sublist in inputs for val in sublist]
	targets = [val for sublist in targets for val in sublist]
	step = lambda x : [int(x+i*window_size/stride) for i in range(batchlen)]

	if shuffle:
		np.random.shuffle(starts)
	batches = np.empty((batchsize,batchlen),dtype=np.int32)
	if num_batches != -1:
		num_batches = int(num_batches*batchsize) # Convert num_batches to number of metabatches.
		if num_batches > len(starts):
			num_batches = -1

	for i,start in enumerate(starts[0:num_batches]):
		batch = np.array([i for i in step(start)],dtype=np.int32)
		batches[i%batchsize] = batch
		if i%batchsize == batchsize-1:
			batches = batches.transpose()
			for pos,batch in enumerate(batches):
				yield np.array([inputs[i] for i in batch]), np.array([targets[i] for i in batch]), pos
				batches = np.empty((batchsize,batchlen),dtype=np.int32)
		if drop_last == False and i==len(starts) and i%batchsize!=0:
			batches = batches[0:i%batchsize]
			batches = batches.transpose()
			for pos,batch in enumerate(batches):
				yield np.array([inputs[i] for i in batch]), np.array([targets[i] for i in batch]), pos


def iterate_minibatches_test(inputs, targets, window_size, stride):
	assert (window_size/stride).is_integer(), 'in order to generate sequential batches, the sliding window length must be divisible by the step.'
	starts = [[(x,int(np.floor(len(i)/window_size))) for x in range(0,window_size)] for i in inputs]
	starts = [sublist for sublist in starts]
	inputs = [sublist for sublist in inputs]
	targets = [sublist for sublist in targets]
	step = lambda x,j : [int(x+i*window_size/stride) for i in range(j)]
	for i in range(len(inputs)):
		for start in starts[i][0:window_size]:
			start,batchlen = start
			batches = np.array([np.array([i for i in step(start[0],start[1])]) for start in starts[i][0:window_size]])
		batches = batches.transpose()			
		for pos,batch in enumerate(batches):
			yield np.array([inputs[i][j] for j in batch]), np.array([targets[i][j] for j in batch]), pos


class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=5, verbose=False, delta=0, path=''):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved.
							Default: 5
			verbose (bool): If True, prints a message for each validation loss improvement. 
							Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement.
							Default: 0
			path (str): Path for the checkpoint to be saved to.
							Default: 'checkpoint.pt'
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.path = path


	def __call__(self, val_loss, model, path):
		self.path = path + '.pt'
		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
		elif score < self.best_score + self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model)
			self.counter = 0


	def save_checkpoint(self, val_loss, model):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		torch.save(model.state_dict(), self.path)
		self.val_loss_min = val_loss


def norm_shape(shape):
	'''
	Normalize numpy array shapes so they're always expressed as a tuple,
	even for one-dimensional shapes.
	Parameters
		shape - an int, or a tuple of ints
	Returns
		a shape tuple
	'''
	try:
		i = int(shape)
		return (i,)
	except TypeError:
		# shape was not a number
		pass
	try:
		t = tuple(shape)
		return t
	except TypeError:
		# shape was not iterable
		pass
	raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a,ws,ss = None,flatten = True):
	'''
	Return a sliding window over a in any number of dimensions
	Parameters:
		a  - an n-dimensional numpy array
		ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
			 of each dimension of the window
		ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
			 amount to slide the window in each dimension. If not specified, it
			 defaults to ws.
		flatten - if True, all slices are flattened, otherwise, there is an
				  extra dimension for each dimension of the input.
	Returns
		an array containing each n-dimensional window from a
	'''

	if None is ss:
		# ss was not provided. the windows will not overlap in any direction.
		ss = ws
	ws = norm_shape(ws)
	ss = norm_shape(ss)

	# convert ws, ss, and a.shape to numpy arrays so that we can do math in every
	# dimension at once.
	ws = np.array(ws)
	ss = np.array(ss)
	shape = np.array(a.shape)


	# ensure that ws, ss, and a.shape all have the same number of dimensions
	ls = [len(shape),len(ws),len(ss)]
	if 1 != len(set(ls)):
		raise ValueError(\
		'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

	# ensure that ws is smaller than a in every dimension
	if np.any(ws > shape):
		raise ValueError(\
		'ws cannot be larger than a in any dimension.\
        a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

	# how many slices will there be in each dimension?
	newshape = norm_shape(((shape - ws) // ss) + 1)
	# the shape of the strided array will be the number of slices in each dimension
	# plus the shape of the window (tuple addition)
	newshape += norm_shape(ws)
	# the strides tuple will be the array's strides multiplied by step size, plus
	# the array's strides (tuple addition)
	newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
	strided = ast(a,shape = newshape,strides = newstrides)
	if not flatten:
		return strided

	# on than the window. I.e., the new array is a flat list of slices.
	meat = len(ws) if ws.shape else 0
	firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
	dim = firstdim + (newshape[-meat:])

	return strided.reshape(dim)


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss