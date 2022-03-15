import os, torch, sys
from utils import *
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


def load_model(model, optimizer, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def moving_average_standardize(W, n):
    T = W.shape[0]
    std_W = (W[:n, :] - W[:n, :].mean())/W[:n, :].std() 
    for i in range(n,T):
        ref = W[i+1-n:i+1, :]
        w = (W[i:i+1, :] - ref.mean())/ref.std()
        std_W = torch.cat((std_W,w))
    return std_W

def basis_function(d, shape):
    m = d.shape[0]
    sorted_d = np.sort(d)
    g = []
    for i in range(m):
        if shape == 'monotone_inc':
            a = (d >= sorted_d[i]).astype('float')
            b = int(sorted_d[i] <= 0.0) 
            g.append(a - b)
        elif shape == 'concave_inc':
            a = (d <= sorted_d[i]).astype('float')
            gx = np.multiply(d-sorted_d[i], a) + sorted_d[i] * int(sorted_d[i] >= 0.0) 
            g.append(gx)

    return np.stack(g, axis=1)

def generate_data(X, p):
    T = X.size(1)
    input, target = [], []
    input_mask = []

    k = T // 1000

    for i in range(p, T):        
        target.append(X[:, i])
        mask = torch.div(torch.arange(i-p, i), 1000, rounding_mode='trunc')
        input_mask.append(torch.eye(k)[mask])
        input.append(X[:, i-p:i])
    return torch.stack(input), torch.stack(target), torch.stack(input_mask), k


class Model(nn.Module):
    def __init__(self, N, g, k):
        """
        k : number of shape functions
        """
        super(Model, self).__init__()

        self.g = g
        self.N = N
        self.k = k

        # Defining some parameters
        w = torch.empty(k, N * N)
        c = torch.empty(N, 1)
        self.weights = nn.Parameter(nn.init.xavier_normal_(w)) # [k, N, 1]
        # self.coefs = nn.Parameter(nn.init.xavier_normal_(c)) # [N, 1]

    def forward(self, x, x_mask):
        """
        x : [b, N, p]
        x_mask : [b, p, k] (one-hot)
        """
        self.g.requires_grad = False
        wg = torch.matmul(self.weights ** 2, self.g) 
        # wg = torch.matmul(weights ** 2, g)

        w = torch.matmul(x_mask, wg)
        p = x.size(-1)
        
        f = w.reshape(-1, p, self.N, self.N) # add exponential term
        ef = torch.exp(-f)
        
        # mu_t_k = self.mu_basis(x_i)
        # x = x - torch.mul(mu_t_k, self.coefs)  # [b, N, p]  
        Z = []
        for b in range(x.size(0)):
            xs = x[b, ].reshape(p, -1, 1)
            z = torch.matmul(ef[b, ], xs)
            Z.append(z.sum(0).squeeze())
        
        Z = torch.stack(Z)

        # mu_t = self.mu_basis(y_i).unsqueeze(-1)        
        # z += torch.mul(mu_t, self.coefs).squeeze(-1)  # [b, N]
        return Z, w
    
    def mu_basis(self, v):
        return v ** 2



def train(X, d, p, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    
    N, T = X.shape   

    # Generate data 
    # input :  [T - 1, N, 1]
    input, target, input_mask, k = generate_data(X, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=False)

    #  Intialize model
    model = Model(N, g, k)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        losses = 0
        for idx in tqdm(loader): 
            y = target[idx, ]
            x = input[idx, ]
            x_mask = input_mask[idx, ]
            # x_i, y_i = input_indices[idx, ], target_indices[idx, ]

            # x_i = torch.repeat_interleave(x_i.unsqueeze(1), N, 1)
            # y_i = torch.repeat_interleave(y_i.unsqueeze(1), N, 1)

            pred, _ = model(x, x_mask)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            # print(loss)
            loss.backward()
            
            optimizer.step()
            losses += loss.item()

        train_loss = losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.3f}"
        print(msg)

        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)

def forecast(X, d, p, model_path, forecast_path, shape, device='cpu'):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    #  Intialize model
    N, _ = X.shape 
    
    model = Model(N, g)
    load_model(model, None, model_path, device)
    model.eval()

    Xts = torch.from_numpy(X).float()
    input, input_indices, target, target_indices = generate_data(Xts, p)
    loss_fn = nn.MSELoss()
    
    print('Predicting ...')
    pred, F = model(input)
    pred = pred.squeeze(-1)
    loss = loss_fn(pred, target)
 
    pred = torch.t(pred)
    out = torch.cat((Xts[:, :1], pred), dim=-1)

    out = out.detach().numpy()

    print(loss)
    if forecast_path:
        write_pickle([X, out, F], forecast_path)
    
    
def scale(X, max_, min_):
    X_std = (X - X.min(axis=1).reshape(-1,1)) / ((X.max(axis=1) - X.min(axis=1)).reshape(-1,1))
    X_std = X_std * (max_ - min_) + min_
    return X_std

if __name__ == "__main__":


    sample_path = 'data/sample.pickle'
    data_path = 'data/nst_sim.npy'
    model_path = 'model/nst_sim.pt'
    forecast_path = 'output/nst_sim.pickle'


    train_size = 3000
    batch_size = 100
    epochs = 1000
    lr = 0.01
    
    p = 1


    data = np.load(data_path)
    _, d, _ = load_pickle(sample_path)

    data_std = scale(data, 1, 0)
    # data_std = data
 

    # d_norm = scale(d.reshape(1,-1), 1, 0).reshape(-1)
    d_norm = d
    
        
    X = data_std[:, :train_size]
    X = torch.from_numpy(X).float()

    shape = 'monotone_inc'

    if sys.argv[1] == 'train':
        train(X, d_norm, p, model_path, batch_size, epochs, lr, shape, device='cpu')
    else:
        forecast(data, d_norm, p, model_path, forecast_path, shape, k)





