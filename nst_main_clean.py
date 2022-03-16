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
    input_indices = []

    k = T // 1000
    target_indices = torch.arange(p, T)

    for i in range(p, T):        
        target.append(X[:, i])
        mask = torch.div(torch.arange(i-p, i), 1000, rounding_mode='trunc')
        input_mask.append(mask)
        input.append(X[:, i-p:i])
        input_indices.append(torch.arange(i-p, i))
    
    return torch.stack(input), torch.stack(target), torch.stack(input_mask), torch.stack(input_indices), target_indices, k


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
        self.weights = nn.Parameter(nn.init.xavier_uniform_(torch.empty(N * N, 1))) # [k, N, 1]
        # self.alphas = nn.Parameter(nn.init.xavier_uniform_(torch.empty(k, 1)))
        self.alphas = torch.tensor([[0.001],[0.002], [0.004]])

    def forward(self, x, x_mask, x_i, y_i):
        """
        x : [b, N, p]
        x_mask : [b, p, k] (one-hot)
        """
        self.g.requires_grad = False
        wg = torch.matmul(self.g, self.weights ** 2) 
        # wg = torch.matmul(g, weights ** 2)
        x_i = self.mu_basis(x_i)
        y_i = self.mu_basis(y_i)

        f = torch.exp(-wg).reshape(self.N, self.N)
        # f = torch.exp(-wg).reshape(N, N)

        Z = []
        for b in range(x.size(0)):
            z = 0
            for i in range(x.size(-1)):
                xs = x[b, :, i:i+1] - x_i[b, i:i+1]
                ef = (x_mask[b, i, ] @ (self.alphas ** 2)) * f
                z += torch.matmul(ef, xs)
            
            z = z + y_i[b]
            Z.append(z)
        
        Z = torch.stack(Z).squeeze(-1)
        return Z, wg
    
    def mu_basis(self, v):
        return 0.1 * v



def train(X, d, p, model_path, batch_size, epochs, lr, shape, device='cpu'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    F = np.log(d+1)
    F = torch.from_numpy(F)
    
    
    N, T = X.shape   

    # Generate data 
    # input :  [T - 1, N, 1]
    input, target, input_mask, input_indices, target_indices, k = generate_data(X, p)
    loader = DataLoader(list(range(T-p)), batch_size=batch_size, shuffle=True)

    #  Intialize model
    model = Model(N, g, k)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    if os.path.isfile(model_path):
        load_model(model, optimizer, model_path, device)
    else:
        model.to(device)

    loss_fn = nn.MSELoss()
    prev_loss = 1.0

    for epoch in range(1, epochs + 1):
        train_losses = 0
        val_losses = 0
        for idx in tqdm(loader): 
            y = target[idx, ]
            x = input[idx, ]
            x_mask = input_mask[idx, ]
            x_mask = torch.eye(k)[x_mask]
    
            x_i, y_i = input_indices[idx, ], target_indices[idx, ]

            pred, W = model(x, x_mask, x_i, y_i)
            W = W.squeeze(-1)
            optimizer.zero_grad()
            loss = loss_fn(pred, y)
            loss.backward()
            
            optimizer.step()
            train_losses += loss.item()
            val_losses += loss_fn(W, F).item()

        train_loss = train_losses / len(loader)
        val_loss = val_losses / len(loader)
        msg = f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Val loss: {val_loss:.5f}"
        print(msg)
        if val_loss < prev_loss:
            print('Saving model ...')
            torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, model_path)
            prev_loss = val_loss

def forecast(X, d, p, model_path, forecast_path, shape, device='cpu'):

    g = basis_function(d, shape)
    g = torch.from_numpy(g).float()
    
    N, _ = X.shape 

    input, target, input_mask, input_indices, target_indices, k = generate_data(X, p)
    
    model = Model(N, g, k)
    load_model(model, None, model_path, device)
    model.eval()

    loss_fn = nn.MSELoss()
    
    print('Predicting ...')
    input_mask = torch.eye(k)[input_mask]
    pred, W = model(input, input_mask, input_indices, target_indices)
    pred = pred.squeeze(-1)
    loss = loss_fn(pred, target)
 
    pred = torch.t(pred)
    out = torch.cat((X[:, :p], pred), dim=-1)

    out = out.detach().numpy()
    W = W.detach().numpy()

    print(loss)
    print(model.alphas ** 2)
    if forecast_path:
        X = X.detach().numpy()
        write_pickle([X, out, W], forecast_path)
    
    
def scale(X, max_, min_):
    X_std = (X - X.min(axis=1).reshape(-1,1)) / ((X.max(axis=1) - X.min(axis=1)).reshape(-1,1))
    X_std = X_std * (max_ - min_) + min_
    return X_std

if __name__ == "__main__":


    sample_path = 'data/sample.pickle'
    data_path = 'data/nst_sim_clean.npy'
    model_path = 'model/nst_sim_clean2.pt'
    forecast_path = 'output/nst_sim_clean.pickle'


    train_size = 3000
    batch_size = 100
    epochs = 50
    lr = 1e-3
    
    p = 1


    data = np.load(data_path)
    _, d, _ = load_pickle(sample_path)

    # data_std = scale(data, 1, 0)
    data_std = data
 

    # d_norm = scale(d.reshape(1,-1), 1, 0).reshape(-1)
    d_norm = d
    
        
    X = data_std[:, :train_size]
    X = torch.from_numpy(X).float()

    shape = 'monotone_inc'

    if sys.argv[1] == 'train':
        train(X, d_norm, p, model_path, batch_size, epochs, lr, shape, device='cpu')
    else:
        forecast(X, d_norm, p, model_path, forecast_path, shape)





