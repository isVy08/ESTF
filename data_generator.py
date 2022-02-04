import torch
import numpy as np
from utils import *


def generate_weight(x, f, method='exp'):
    N = x.shape[0]
    fd = f.reshape((N, N))
    if method == 'static':
        w = np.true_divide(1, fd**2)
        np.fill_diagonal(w, 0)
    else: 
        
        if method == 'exp':
            fd = np.exp(-fd)
            w = fd/np.sum(fd, axis=0)
        elif method == 'invsqr':
            fd = 1/fd**2
            w = fd/np.sum(fd, axis=0)
        elif method == 'original':
            w = 1 - fd/np.sum(fd, axis=0)
            w = w / (N-1)
        
    return torch.from_numpy(w @ x).float() 

def generate_weight_batch(X, F, method='exp'):
    W = []
    T = X.shape[1]
    
    for t in range(T):

        if method == 'static':
            z = generate_weight(X[:,t], F, method) # F := D
        else:       
            z = generate_weight(X[:,t], F[:,t], method)
            
        W.append(z)

    return torch.stack(W).float()    

def moving_average_standardize(W, n):
    T = W.shape[0]
    std_W = (W[:n, :] - W[:n, :].mean())/W[:n, :].std() 
    for i in range(n,T):
        ref = W[i+1-n:i+1, :]
        w = (W[i:i+1, :] - ref.mean())/ref.std()
        std_W = torch.cat((std_W,w))
    return std_W

def generate_input(W, agg_W, std_W, p, s):
    """
    Generate input for the next observation
    W is raw weights
    """ 
    # Aggregated weight for the next observation
    w = W[-p:,].sum(dim=0).unsqueeze(0)
    agg_W = torch.cat((agg_W, w))

    # Standardize w by p moving average on agg_W
    std_w = (w - agg_W[-s:,].mean())/agg_W[-s:,].std()
    std_W = torch.cat((std_W, std_w))
    return std_W[-p:,].unsqueeze(0), agg_W, std_W

class DataGenerator():
    def __init__(self, config):
        super(DataGenerator).__init__()
        """
        F : f(D_ij) ( N * N x T)
        X : ( N x T)
        """
        X = np.load(config.data_path)
        X = X[:, :config.train_size]
        
        self.N, self.T = X.shape
        self.p = config.window_size
        self.h = config.holdout_size
        self.s = config.slide_size
            
        if config.weight_method == 'static':
            _, F, _ = load_pickle(config.sample_path)
        else:
            F = np.load(config.estimate_path) 
            F = F[:, :config.train_size]

        self.W = generate_weight_batch(X, F, config.weight_method)   
        self.agg_W = self.transform(self.W)
        
        self.std_W = moving_average_standardize(self.agg_W, self.s)
        self.Z = self.generate_input(self.std_W) # (T x p x N)
        self.train_Z, self.test_Z = self.split_train_test(self.Z)
        
        
        input_X = torch.from_numpy(X.transpose()).float()
        self.X = self.generate_label(input_X)
        self.train_X, self.test_X = self.split_train_test(self.X)
        
    def transform(self, W):
        T, N = W.shape
        A = W[:1, :]
        A = torch.zeros((1, N))
        for i in range(1, T):
            a = W[:i,:] if i < self.p else W[(i-self.p):i,:]
            a = a.sum(dim=0).unsqueeze(dim=0)
            A = torch.cat((A, a), dim=0)
        return A
    
    def generate_input(self, W):

        Z = W[self.p:(self.p*2), :].unsqueeze(dim=0)
        for i in range(self.p*2+1, self.T):
            z = W[(i-self.p):i, :].unsqueeze(dim=0)
            Z = torch.cat((Z, z), dim=0)
        return Z
    
    def split_train_test(self, input):
        return input[:-self.h, ], input[-self.h:,]
    
    def generate_label(self, input_X):
        L = []
        T = input_X.shape[0]
        for i in range(self.p*2, T):
            l = input_X[(i-self.p+1):(i+1), ]
            L.append(l)
        return torch.stack(L)

        

    
    

