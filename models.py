import torch
from torch import nn

class DeepConvLSTM(nn.Module):

    def __init__(self, n_channels, len_seq=24, n_hidden=128, n_layers=2, n_filters=64,
                n_classes=18, filter_size=5, pool_filter_size=3, dropout_probability=0.5):
        super(DeepConvLSTM, self).__init__() # Call init function for nn.Module whenever this function is called
        self.n_channels = n_channels
        self.dropout_probability = dropout_probability # Dropout probability
        self.len_seq = len_seq 
        self.n_layers = n_layers # Number of layers in the lstm network
        self.n_hidden = n_hidden # number of hidden units per layer in the lstm
        self.n_filters = n_filters # number of convolutional filters per layer
        self.n_classes = n_classes # number of target classes
        self.filter_size = filter_size # convolutional filter size
        self.pool_filter_size = pool_filter_size # max pool filter size if using

        self.convlayer = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, (filter_size)),          
            nn.Conv1d(n_filters, n_filters, (filter_size)),          
            nn.Conv1d(n_filters, n_filters, (filter_size)),
            nn.Conv1d(n_filters, n_filters, (filter_size))
            )
        self.lstm = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.predictor = nn.Linear(n_hidden,n_classes)


    def forward(self, x, hidden, batch_size):
        x = x.view(-1, self.n_channels, self.len_seq)
        x = self.convlayer(x)
        x = x.view(batch_size, -1, self.n_filters)
        x,hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.view(batch_size, -1, self.n_hidden)[:,-1,:]
        out = self.predictor(x)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data # return a Tensor from self.parameters to use as a base for the initial hidden state.
        ## Generate new tensors of zeros with similar type to weight, but different size.
        if (torch.cuda.is_available()):
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(), # Hidden state
                  weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda()) # Cell state
        else:
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                      weight.new_zeros(self.n_layers, batch_size, self.n_hidden))

        return hidden


class DeepConvLSTM_mod(nn.Module):

    def __init__(self, n_channels, len_seq=24, n_hidden = 128, n_layers = 2, n_filters = 64,
                n_classes = 2, filter_size = 5,pool_filter_size=3, dropout_probability = 0.5):
        super(DeepConvLSTM_mod, self).__init__() # Call init function for nn.Module whenever this function is called
        self.n_channels = n_channels
        self.dropout_probability = dropout_probability # Dropout probability
        self.len_seq = len_seq 
        self.n_layers = n_layers # Number of layers in the lstm network
        self.n_hidden = n_hidden # number of hidden units per layer in the lstm
        self.n_filters = n_filters # number of convolutional filters per layer
        self.n_classes = n_classes # number of target classes
        self.filter_size = filter_size # convolutional filter size
        self.pool_filter_size = pool_filter_size # max pool filter size if using

        # Convolutional net
        self.convlayer = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, (filter_size)),
            nn.Conv1d(n_filters, n_filters, (filter_size)),
            nn.Conv1d(n_filters, n_filters, (filter_size)),
            nn.Conv1d(n_filters, n_filters, (filter_size))
            )
        # LSTM layers
        self.lstm = nn.LSTM(n_filters, n_hidden, n_layers, batch_first=True)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout_probability)
        # Output layer
        self.predictor = nn.Linear(n_hidden,n_classes)


    def forward(self, x, hidden, batch_size):

        #Reshape x if necessary to add the 2nd dimension
        x = x.view(-1, self.n_channels, self.len_seq)
        x = self.convlayer(x)
        x = x.view(batch_size, -1, self.n_filters)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.view(batch_size, -1, self.n_hidden)[:,-1,:]
        out = self.predictor(x)

        return out, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data # return a Tensor from self.parameters to use as a base for the initial hidden state.
        ## Generate new tensors of zeros with similar type to weight, but different size.
        if (torch.cuda.is_available()):
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda(), # Hidden state
                  weight.new_zeros(self.n_layers, batch_size, self.n_hidden).cuda()) # Cell state
        else:
            hidden = (weight.new_zeros(self.n_layers, batch_size, self.n_hidden),
                      weight.new_zeros(self.n_layers, batch_size, self.n_hidden))

        return hidden