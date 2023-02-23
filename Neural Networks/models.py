# import required packages
import torch
from torch import nn

# basic Feed Forward Neural Network
class FFNetwork(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions, output_dimensions):
        super(FFNetwork, self).__init__()

        self.linear1 = nn.Linear(input_dimensions, hidden_dimensions)

        self.nonlinear = nn.ReLU()
    
        self.linear2 = nn.Linear(hidden_dimensions, output_dimensions)

        self.sigmoid = torch.sigmoid
    
    def forward(self, x):

        x = self.linear1(x)

        x = self.nonlinear(x)

        output = self.linear2(x)

        output = self.sigmoid(output)
        return output


# basic Feed Forward Neural Network (Regression)
class FFNetworkReg(nn.Module):
    def __init__(self, input_dimensions, hidden_dimensions, output_dimensions):
        super(FFNetworkReg, self).__init__()

        self.linear1 = nn.Linear(input_dimensions, hidden_dimensions)

        self.nonlinear = nn.ReLU()
    
        self.linear2 = nn.Linear(hidden_dimensions, output_dimensions)
    
    def forward(self, x):

        x = self.linear1(x)

        x = self.nonlinear(x)

        output = self.linear2(x)

        return output


# Define Fully Connected FF network for the Embedding model
class FFNetworkEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(FFNetworkEmbedding, self).__init__()

        self.embed = nn.EmbeddingBag(input_dim, embedding_dim)

        self.hidden = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.3)
        

    def forward(self, x, offsets):

        # apply an emedding bag layer to get average of all embeddings
        embeddings = self.embed(x, offsets)

        # apply linear functions
        output = self.hidden(embeddings)

        # apply dropout to avoid overfitting
        output = self.dropout(output)

        return output, embeddings



# Define Fully Connected FF network for the Embedding model
class FFNetworkEmbedding2(nn.Module):
    def __init__(self, input_dim, embedding_dim, output_dim):
        super(FFNetworkEmbedding2, self).__init__()

        self.embed = nn.EmbeddingBag(input_dim, embedding_dim)

        self.hidden = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.3)
        

    def forward(self, x, offsets):

        # apply an emedding bag layer to get average of all embeddings
        embeddings = self.embed(x, offsets)

        # apply linear functions
        output = self.hidden(embeddings)

        # apply dropout to avoid overfitting
        output = self.dropout(output)

        return output, embeddings




# define Fully Connected Network which uses bert embeddings for titles
class FFNetworkBertEmbedding(nn.Module):

    def __init__(self, output_dim, embedding_dim = 768):
        super(FFNetworkBertEmbedding, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 900),
            nn.ReLU(),

            nn.Linear(900, 800),
            nn.ReLU(),

            nn.Linear(800, 700),
            nn.ReLU(),

            nn.Linear(700, 600),
            nn.ReLU(),

            nn.Linear(600, 300),
            nn.ReLU(),

            nn.Linear(300, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim)
        )

        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        
        output = self.hidden(x)

        #output = self.dropout(output)

        return output
        


class FFNonEmbedding(nn.Module):
    def __init__(self, output_dim, input_dim = 7):
        super(FFNonEmbedding, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

        #self.dropout = nn.Dropout(0.6)

    def forward(self, x):
        
        output = self.hidden(x)

        #output = self.dropout(output)

        return output



# define Fully Connected Network which uses bert embeddings for titles
class FFNetworkBertEmbeddingMultiClass(nn.Module):

    def __init__(self, output_dim, embedding_dim = 768):
        super(FFNetworkBertEmbeddingMultiClass, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(embedding_dim, 2048),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.ReLU(),

            nn.Linear(1024, 900),
            nn.ReLU(),

            nn.Linear(900, 800),
            nn.ReLU(),

            nn.Linear(800, 700),
            nn.ReLU(),

            nn.Linear(700, 600),
            nn.ReLU(),

            nn.Linear(600, 300),
            nn.ReLU(),

            nn.Linear(300, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim)
        )

        self.softmax = nn.Softmax()

        #self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        
        output = self.hidden(x)
        output = self.softmax(output)

        #output = self.dropout(output)

        return output
        

# Define Recurrent Neural Network
class RNN(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, output_dimension, num_hidden):
        super(RNN, self).__init__()

        # track dimensions used throughout
        self.hidden_dim = hidden_dimension
        self.num_hidden = num_hidden

        # define RNN layers
        self.rnn = nn.RNN(input_dimension, hidden_dimension, num_hidden, batch_first=True, nonlinearity='relu')

        # define Readoutlayer
        self.out = nn.Linear(hidden_dimension, output_dimension)

    def intilize_hidden(self, batch_size):
        hidden_state = torch.zeros(self.num_hidden, self.hidden_dim).requires_grad_()
        return hidden_state

    def forward(self, x):

        # want batch size for initial hidden state
        batch_size = x.size(0)

        # intilaize hidden state
        hidden = self.intilize_hidden(batch_size)

        # apply rnn 
        output, hidden = self.rnn(x, hidden)

        # apply readout
        output = self.out(output)

        # return output predictions and resultant hidden state
        return output


