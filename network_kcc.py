import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
class KccRNN(nn.Module):
    
    def __init__(self, tokens,embedding_dim=50, n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.kccs = tokens
        self.int2kcc = dict(enumerate(self.kccs ))
        self.kccs2int = {ch: ii for ii, ch in self.int2kcc.items()}
        
        self.word_embeddings = nn.Embedding(len(self.kccs), embedding_dim)

        ## TODO: define the LSTM
        self.lstm = nn.LSTM(embedding_dim, n_hidden, n_layers, 
                            dropout=drop_prob, batch_first=True,  bidirectional=True)
        
        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(n_hidden*2, 1)
      
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
        embeds = self.word_embeddings(x)

        ## TODO: Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(embeds, hidden)
        
        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)
        
        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden*2)
        
        ## TODO: put x through the fully-connected layer
        out = self.fc(out)
        
        # return the final output and the hidden state
        return out.squeeze(), hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        train_on_gpu = torch.cuda.is_available()
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers*2, batch_size, self.n_hidden).zero_())
        
        return hidden