import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

#https://github.com/pytorch/pytorch/issues/36591
class linear_with_channels(nn.Module):

    def __init__(self, input_dim, output_dim, num_channels):
        super().__init__()
        
        self.w = nn.Parameter(torch.zeros(num_channels, input_dim, output_dim))
        # with torch.no_grad():
        #     self.w[1,:,:] = torch.ones(input_dim, output_dim)
        self.b = nn.Parameter(torch.zeros(num_channels, 1, output_dim))

    def initialize_params(self, weight_initialization):
        if weight_initialization == "He":
            torch.nn.init.kaiming_uniform_(self.w)
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w)
            bound = 1. / np.sqrt(fan_in)
            torch.nn.init.uniform_(self.b, -bound, bound)
        elif weight_initialization == "normal":
            nn.init.normal_(self.w, mean=0., std=0.1)
            nn.init.zeros_(self.b)
        elif weight_initialization == "xavier":
            nn.init.xavier_normal_(self.w)
        else: # if scheme is invalid, use normal initialization as a substitute
            nn.init.normal_(self.w, mean=0., std=0.1)
            nn.init.zeros_(self.w)

    def forward(self, X):
        
        # [b, c, in] x [b, in, out] = [b, c, out]
        # [c, b, in] x [c, in, out] = [c, b, out]
        X = X.permute(1, 0, 2) # <- [batch, channels, in] -> [channels, batch, in]
        X = torch.bmm(X, self.w) + self.b
        X = X.permute(1, 0, 2) # <- [channels, batch, out] -> [batch, channels, out]
        return X

class block_resmlp(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers,
                 skip_connection):
        
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module("layer0",    nn.Linear(input_dim, output_dim))
        self.layers.add_module("ReLU", nn.ReLU())
        for i in range(num_layers-1):
            self.layers.add_module("layer"+str(i+1), nn.Linear(output_dim, output_dim))
            self.layers.add_module("ReLU",      nn.ReLU())
    
        if skip_connection:
            self.skip_layer = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        Y = self.layers(X)
        return Y + self.skip_layer(X)

class block_resnet(nn.Module):
    
    def __init__(self, input_dim:int, output_dim:int, num_layers:int, skip_connection:bool=True):
        """Initializes a resnet MLP block

        Args:
            input_dim (int): input dimension.
            output_dim (int): output dimension. All layers except for the first one will have this dimension
            num_layers (int): numer of layers to include in the block. Except for the first layer, will all have shape (output_dim, output_dim)
            skip_connection (bool, optional): whether to include a redidual connection, where the input is add
                to the output. Defaults to True.
        """
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module("layer0",    nn.Linear(input_dim, output_dim))
        self.layers.add_module("Activation", activation_function(output_dim))
        for i in range(num_layers-1):
            self.layers.add_module("layer"+str(i+1), nn.Linear(output_dim, output_dim))
            self.layers.add_module("bn"+str(i+1),    nn.BatchNorm1d(output_dim))
            self.layers.add_module("Activation",     activation_function(output_dim))
    
        if skip_connection:
            self.skip_layer = nn.Linear(input_dim, output_dim)
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, X):
        Y = self.layers(X)
        return Y + self.bn(self.skip_layer(X))

class block_parallel_resnet(nn.Module):
    
    def __init__(self, input_dim, output_dim, num_layers, num_channels, skip_connection):
        
        super().__init__()

        self.layers = nn.Sequential()
        self.layers.add_module("layer0",  linear_with_channels(input_dim, output_dim, num_channels))
        self.layers.add_module("ReLU",    nn.ReLU())
        for i in range(num_layers-1):
            self.layers.add_module("layer"+str(i+1), linear_with_channels(output_dim, output_dim, num_channels))
            self.layers.add_module("bn"+str(i+1),    nn.BatchNorm1d(num_channels))
            self.layers.add_module("ReLU",           nn.ReLU())
    
        if skip_connection:
            self.skip_layer = linear_with_channels(input_dim, output_dim, num_channels)
            self.bn = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = self.layers(X)
        return Y + self.bn(self.skip_layer(X))


class multi_headed_attention(nn.Module):

    def __init__(self, hidden_dim, num_heads=2, dropout_prob=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.layer_q = nn.Linear(hidden_dim, hidden_dim)
        self.layer_k = nn.Linear(hidden_dim, hidden_dim)
        self.layer_v = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_prob)
        #self.softmax = nn.Softmax(hidden_dim)

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads."""
        # Shape of input X: (batch_size, no. of queries or key-value pairs, num_hiddens). 
        # Shape of output X: (batch_size, no. of queries or key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)
        X = X.permute(0, 2, 1, 3)

        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

    def dot_product_attention(self, q, k, v):
        dim = q.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2)
        # calculate attention scores using the dot product
        scores = torch.bmm(q, k.transpose(1, 2)) / np.sqrt(dim)
        # normalize so that sum(scores) = 1 and all scores > 0
        #self.attention_weights = masked_softmax(scores, valid_lens)
        self.attention_weights = F.softmax(scores, dim=-1)
        # perform a batch matrix multiplaction to get the attention weights
        return torch.bmm(self.dropout(self.attention_weights), v)

    def forward(self, queries, keys, values):

        queries = self.transpose_qkv(self.layer_q(queries))
        keys    = self.transpose_qkv(self.layer_k(keys))
        values  = self.transpose_qkv(self.layer_v(values))

        X = self.dot_product_attention(queries, keys, values)
        X = X.reshape(-1, self.hidden_dim)
        X = self.out(X)
        return X

class block_addnorm(nn.Module):
    def __init__(self, shape, dropout_prob=0.):
        super().__init__()
        self.dropoiut = nn.Dropout(dropout_prob)
        self.layerNorm = nn.LayerNorm(shape)
    def forward(self, X, Y):
        return self.layerNorm(self.dropoiut(Y) + X)
    
class block_transformer_encoder(nn.Module):
    """
    """

    def __init__(self, hidden_dim, num_channels, dropout_prob=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_channels = num_channels

        #self.ln1 = nn.LayerNorm(self.hidden_dim)
        #self.attention = multi_headed_attention(self.hidden_dim, 1, dropout_prob)
        self.attention = nn.MultiheadAttention(int(self.hidden_dim), 1, dropout_prob, batch_first=True)
        #self.addnorm1 = block_addnorm(self.hidden_dim, dropout_prob)

        #feed-forward network
        #self.ln2 = nn.LayerNorm(self.hidden_dim)
        #self.h1 = nn.Linear(int(self.hidden_dim / num_channels), int(self.hidden_dim / num_channels))
        self.h1 = linear_with_channels(int(self.hidden_dim/num_channels), int(self.hidden_dim/num_channels), num_channels)
        self.activation = activation_function(int(self.hidden_dim/num_channels))
        #self.addnorm2 = block_addnorm(self.hidden_dim, dropout_prob)

    def forward(self, X):
        X = torch.unsqueeze(X, 1)
        X = X + self.attention(X, X, X)[0]
        X = X.reshape(-1, X.shape[2])

        Y = X.reshape(-1, self.num_channels, int(self.hidden_dim / self.num_channels))
        Y = self.activation(self.h1(Y))
        X = X + Y.reshape(-1, self.hidden_dim)
        return X

class activation_function(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.dim = d
        self.gamma = nn.Parameter(torch.zeros(d))
        self.beta = nn.Parameter(torch.zeros(d))

    def forward(self, X):
        inv = torch.special.expit(torch.mul(self.beta, X))

        return torch.mul(self.gamma + torch.mul(inv, 1-self.gamma), X)
